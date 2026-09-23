import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import av
except ImportError:  # pragma: no cover - exercised only on environments without PyAV
    av = None


JOINT_ORDER = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


class ASLHand2EgoStereoDataset(Dataset):
    """ASLHand2 ego stereo clips paired with synchronized 3D hand keypoints."""

    def __init__(
        self,
        keypoint_root,
        zed_root,
        sequence_ids=None,
        participants=None,
        split=None,
        split_ratios=(0.7, 0.15, 0.15),
        clip_len=8,
        stride=1,
        image_size=224,
        max_windows=None,
        max_timestamp_mismatch_ms=50.0,
        cache_segments=2,
    ):
        self.keypoint_root = Path(keypoint_root)
        self.zed_root = Path(zed_root)
        self.clip_len = clip_len
        self.stride = stride
        self.image_size = image_size
        self.max_timestamp_mismatch_ms = max_timestamp_mismatch_ms
        self.cache_segments = max(cache_segments, 0)
        self._video_cache = OrderedDict()

        if not self.keypoint_root.exists():
            raise FileNotFoundError(f"ASLHand2 keypoint root not found: {self.keypoint_root}")
        if not self.zed_root.exists():
            raise FileNotFoundError(f"ASLHand2 ZED root not found: {self.zed_root}")

        all_sequences = discover_aslhand2_sequences(self.keypoint_root, self.zed_root)
        selected_sequences = filter_sequences(all_sequences, sequence_ids, participants, split, split_ratios)
        self.participants = sorted({participant_from_sequence(s) for s in selected_sequences})
        self.segments = self._discover_segments(selected_sequences)
        self.windows = self._build_windows()
        if max_windows:
            self.windows = self.windows[:max_windows]
        if not self.windows:
            raise RuntimeError("No ASLHand2 windows found for the requested split/sequences.")

        self.timestamp_stats = self._compute_timestamp_stats()
        print(
            f"ASLHand2 {split or 'custom'}: {len(selected_sequences)} sequences, "
            f"{len(self.segments)} segments, {len(self.windows)} windows, participants={self.participants}"
        )
        print(format_timestamp_stats(self.timestamp_stats))

    def _discover_segments(self, sequence_ids):
        segments = {}
        for sequence_id in sequence_ids:
            keypoint_dir = self.keypoint_root / sequence_id / "keypoints_label"
            zed_dir = self.zed_root / sequence_id
            for kp_path in sorted(keypoint_dir.glob("segment_*.json")):
                seg_idx = kp_path.stem.split("_")[-1]
                left_video = zed_dir / "left" / f"left_segment_{seg_idx}.mp4"
                right_video = zed_dir / "right" / f"right_segment_{seg_idx}.mp4"
                if not left_video.exists() or not right_video.exists():
                    continue
                with open(kp_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                frames = [fr for fr in meta.get("frames", []) if has_both_hands(fr)]
                if len(frames) < self.clip_len:
                    continue
                key = f"{sequence_id}:{seg_idx}"
                segments[key] = {
                    "sequence_id": sequence_id,
                    "segment_id": seg_idx,
                    "participant_id": participant_from_sequence(sequence_id),
                    "left_video": left_video,
                    "right_video": right_video,
                    "frames": frames,
                    "start_timestamp_ms": float(meta.get("start_timestamp_ms", frames[0].get("timestamp_ms", 0.0))),
                    "sentence": meta.get("sentence", ""),
                }
        if not segments:
            raise RuntimeError("No synchronized ASLHand2 segment triplets found.")
        return segments

    def _build_windows(self):
        windows = []
        for seg_key, meta in self.segments.items():
            frames, mismatches = self._match_segment(meta)
            valid = np.asarray([fr.get("rgb_valid", False) for fr in frames], dtype=bool)
            for start in range(0, len(frames) - self.clip_len + 1, self.stride):
                sl = slice(start, start + self.clip_len)
                if valid[sl].all():
                    windows.append((seg_key, start))
            meta["matched_frames"] = frames
            meta["timestamp_mismatches_ms"] = mismatches
        return windows

    def _match_segment(self, meta):
        left_ts = read_video_timestamps_ms(meta["left_video"])
        right_ts = read_video_timestamps_ms(meta["right_video"])
        frames, mismatches = [], []
        for fr in meta["frames"]:
            kp_ts = float(fr["timestamp_ms"])
            video_ts = kp_ts - meta["start_timestamp_ms"] + float(left_ts[0])
            li, lm = nearest_timestamp_index(left_ts, video_ts)
            video_ts = kp_ts - meta["start_timestamp_ms"] + float(right_ts[0])
            ri, rm = nearest_timestamp_index(right_ts, video_ts)
            mismatch = max(lm, rm)
            item = dict(fr)
            item["left_rgb_index"] = li
            item["right_rgb_index"] = ri
            item["timestamp_mismatch_ms"] = mismatch
            item["rgb_valid"] = mismatch <= self.max_timestamp_mismatch_ms
            frames.append(item)
            mismatches.append(mismatch)
        return frames, mismatches

    def _compute_timestamp_stats(self):
        mismatches, skipped, total = [], 0, 0
        for meta in self.segments.values():
            for fr in meta["matched_frames"]:
                total += 1
                mismatches.append(fr["timestamp_mismatch_ms"])
                skipped += int(not fr["rgb_valid"])
        mean = float(np.mean(mismatches)) if mismatches else 0.0
        max_v = float(np.max(mismatches)) if mismatches else 0.0
        return {"mean_ms": mean, "max_ms": max_v, "skipped": skipped, "total": total, "threshold_ms": self.max_timestamp_mismatch_ms}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seg_key, start = self.windows[idx]
        meta = self.segments[seg_key]
        frames = meta["matched_frames"][start:start + self.clip_len]
        left_video, _ = self._read_cached_video(meta["left_video"])
        right_video, _ = self._read_cached_video(meta["right_video"])

        left_imgs, right_imgs, left_kps, right_kps, frame_ids, timestamps, mismatches = [], [], [], [], [], [], []
        for fr in frames:
            left_imgs.append(left_video[fr["left_rgb_index"]])
            right_imgs.append(right_video[fr["right_rgb_index"]])
            left_kps.append(torch.tensor(hand_to_array(fr["hands"]["left"]), dtype=torch.float32))
            right_kps.append(torch.tensor(hand_to_array(fr["hands"]["right"]), dtype=torch.float32))
            frame_ids.append(int(fr.get("frame", len(frame_ids))))
            timestamps.append(float(fr["timestamp_ms"]))
            mismatches.append(float(fr["timestamp_mismatch_ms"]))

        return {
            "left_img_clip": torch.stack(left_imgs),
            "right_img_clip": torch.stack(right_imgs),
            "left_landmarks_clip": torch.stack(left_kps),
            "right_landmarks_clip": torch.stack(right_kps),
            "sequence_id": meta["sequence_id"],
            "participant_id": meta["participant_id"],
            "segment_id": meta["segment_id"],
            "sentence": meta["sentence"],
            "frame_ids": torch.tensor(frame_ids, dtype=torch.long),
            "timestamp_ms": torch.tensor(timestamps, dtype=torch.float32),
            "timestamp_mismatch_ms": torch.tensor(mismatches, dtype=torch.float32),
        }

    def _read_cached_video(self, path):
        key = str(path)
        if key in self._video_cache:
            self._video_cache.move_to_end(key)
            return self._video_cache[key]
        value = read_video_with_timestamps(path, self.image_size)
        if self.cache_segments:
            self._video_cache[key] = value
            while len(self._video_cache) > self.cache_segments:
                self._video_cache.popitem(last=False)
        return value


def discover_aslhand2_sequences(keypoint_root, zed_root):
    sequences = []
    for keypoint_dir in sorted(Path(keypoint_root).iterdir()):
        if not keypoint_dir.is_dir():
            continue
        sequence_id = keypoint_dir.name
        if (keypoint_dir / "keypoints_label").exists() and (Path(zed_root) / sequence_id / "left").exists():
            sequences.append(sequence_id)
    return sequences


def filter_sequences(all_sequences, sequence_ids=None, participants=None, split=None, split_ratios=(0.7, 0.15, 0.15)):
    if sequence_ids:
        requested = parse_csv_arg(sequence_ids)
        selected = [s for s in all_sequences if s in requested]
    else:
        selected = list(all_sequences)
    if participants:
        allowed = set(parse_csv_arg(participants))
        selected = [s for s in selected if participant_from_sequence(s) in allowed]
    if split:
        split_participants = participant_split(sorted({participant_from_sequence(s) for s in selected}), split_ratios)
        allowed = set(split_participants[split])
        selected = [s for s in selected if participant_from_sequence(s) in allowed]
    return selected


def participant_split(participants, ratios=(0.7, 0.15, 0.15)):
    participants = sorted(participants)
    n = len(participants)
    n_train = max(1, int(round(n * ratios[0]))) if n else 0
    n_val = max(1, int(round(n * ratios[1]))) if n >= 3 else max(0, n - n_train)
    if n_train + n_val >= n and n > 1:
        n_val = 1
        n_train = max(1, n - 2)
    return {
        "train": participants[:n_train],
        "val": participants[n_train:n_train + n_val],
        "test": participants[n_train + n_val:],
    }


def participant_from_sequence(sequence_id):
    return sequence_id.split("_")[0]


def parse_csv_arg(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [v.strip() for v in str(value).split(",") if v.strip()]


def has_both_hands(frame):
    hands = frame.get("hands", {})
    return isinstance(hands.get("left"), dict) and isinstance(hands.get("right"), dict)


def hand_to_array(hand):
    return np.asarray([hand[name] for name in JOINT_ORDER], dtype=np.float32)


def read_video_timestamps_ms(path):
    if av is None:
        raise ImportError("PyAV is required for timestamp-accurate ASLHand2 loading. Install with `pip install av`.")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        timestamps = []
        for frame in container.decode(stream):
            if frame.pts is None:
                raise RuntimeError(f"Video frame without PTS in {path}")
            timestamps.append(float(frame.pts * stream.time_base * 1000.0))
    if not timestamps:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.asarray(timestamps, dtype=np.float64)


def read_video_with_timestamps(path, image_size):
    if av is None:
        raise ImportError("PyAV is required for timestamp-accurate ASLHand2 loading. Install with `pip install av`.")
    frames, timestamps = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                raise RuntimeError(f"Video frame without PTS in {path}")
            timestamps.append(float(frame.pts * stream.time_base * 1000.0))
            arr = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)
    video = torch.stack(frames)
    if video.shape[-2:] != (image_size, image_size):
        video = F.interpolate(video, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return video, np.asarray(timestamps, dtype=np.float64)


def nearest_timestamp_index(timestamps_ms, keypoint_timestamp_ms):
    idx = int(np.abs(timestamps_ms - keypoint_timestamp_ms).argmin())
    return idx, float(abs(timestamps_ms[idx] - keypoint_timestamp_ms))


def format_timestamp_stats(stats):
    pct = 100.0 * stats["skipped"] / max(stats["total"], 1)
    return (
        "ASLHand2 RGB-keypoint timestamp matching: "
        f"mean={stats['mean_ms']:.2f} ms, max={stats['max_ms']:.2f} ms, "
        f"skipped={stats['skipped']}/{stats['total']} ({pct:.2f}%), "
        f"threshold={stats['threshold_ms']:.2f} ms"
    )


def default_aslhand2_roots():
    return {
        "keypoint_root": os.environ.get("ASLHAND2_KEYPOINT_ROOT", "/home/jqu11/bigdata/data/ASLHand2/hand_keypoints_synced"),
        "zed_root": os.environ.get("ASLHAND2_ZED_ROOT", "/home/jqu11/bigdata/data/ASLHand2/ZED_Segments"),
    }
