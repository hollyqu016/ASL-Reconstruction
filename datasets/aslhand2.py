import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_video


JOINT_ORDER = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


class ASLHand2EgoStereoDataset(Dataset):
    """ASLHand2 ego stereo clips paired with synchronized 3D hand keypoints.

    The ZED left/right videos are treated as egocentric stereo views. The loader
    only emits samples where a segment JSON and both segment videos exist.
    """

    def __init__(
        self,
        keypoint_root,
        zed_root,
        sequence_id="Abdul_03_52",
        clip_len=8,
        stride=1,
        image_size=224,
        max_windows=None,
    ):
        self.keypoint_dir = Path(keypoint_root) / sequence_id / "keypoints_label"
        self.zed_dir = Path(zed_root) / sequence_id
        self.sequence_id = sequence_id
        self.clip_len = clip_len
        self.stride = stride
        self.image_size = image_size

        if not self.keypoint_dir.exists():
            raise FileNotFoundError(f"ASLHand2 keypoint directory not found: {self.keypoint_dir}")
        if not self.zed_dir.exists():
            raise FileNotFoundError(f"ASLHand2 ZED directory not found: {self.zed_dir}")

        self.segments = self._discover_segments()
        self.windows = []
        for seg_idx, meta in self.segments.items():
            n = len(meta["frames"])
            self.windows.extend((seg_idx, s) for s in range(0, n - clip_len + 1, stride))
        if max_windows:
            self.windows = self.windows[:max_windows]
        if not self.windows:
            raise RuntimeError(f"No ASLHand2 windows found for {sequence_id} with clip_len={clip_len}.")

        print(
            f"ASLHand2 ego stereo {sequence_id}: {len(self.segments)} segments, "
            f"{len(self.windows)} windows, left/right ego views, 21 joints."
        )

    def _discover_segments(self):
        segments = {}
        for kp_path in sorted(self.keypoint_dir.glob("segment_*.json")):
            seg_idx = kp_path.stem.split("_")[-1]
            left_video = self.zed_dir / "left" / f"left_segment_{seg_idx}.mp4"
            right_video = self.zed_dir / "right" / f"right_segment_{seg_idx}.mp4"
            if not left_video.exists() or not right_video.exists():
                continue
            with open(kp_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            frames = [fr for fr in meta.get("frames", []) if has_both_hands(fr)]
            if len(frames) < self.clip_len:
                continue
            segments[seg_idx] = {
                "keypoint_path": kp_path,
                "left_video": left_video,
                "right_video": right_video,
                "frames": frames,
                "start_timestamp_ms": float(meta.get("start_timestamp_ms", frames[0].get("timestamp_ms", 0))),
                "end_timestamp_ms": float(meta.get("end_timestamp_ms", frames[-1].get("timestamp_ms", 1))),
                "sentence": meta.get("sentence", ""),
            }
        if not segments:
            raise RuntimeError(f"No synchronized ASLHand2 triplets found under {self.keypoint_dir} and {self.zed_dir}")
        return segments

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seg_idx, start = self.windows[idx]
        meta = self.segments[seg_idx]
        frames = meta["frames"][start:start + self.clip_len]

        left_video = _read_rgb_video(meta["left_video"], self.image_size)
        right_video = _read_rgb_video(meta["right_video"], self.image_size)
        duration = max(meta["end_timestamp_ms"] - meta["start_timestamp_ms"], 1e-6)

        left_imgs, right_imgs, left_kps, right_kps, frame_ids, timestamps = [], [], [], [], [], []
        for fr in frames:
            ts = float(fr["timestamp_ms"])
            rel = (ts - meta["start_timestamp_ms"]) / duration
            li = _timestamp_to_index(rel, left_video.shape[0])
            ri = _timestamp_to_index(rel, right_video.shape[0])
            left_imgs.append(left_video[li])
            right_imgs.append(right_video[ri])
            left_kps.append(torch.tensor(hand_to_array(fr["hands"]["left"]), dtype=torch.float32))
            right_kps.append(torch.tensor(hand_to_array(fr["hands"]["right"]), dtype=torch.float32))
            frame_ids.append(int(fr.get("frame", len(frame_ids))))
            timestamps.append(ts)

        left_img_clip = torch.stack(left_imgs)
        right_img_clip = torch.stack(right_imgs)
        return {
            "left_img_clip": left_img_clip,
            "right_img_clip": right_img_clip,
            "left_landmarks_clip": torch.stack(left_kps),
            "right_landmarks_clip": torch.stack(right_kps),
            "intrinsics_clip": default_intrinsics(self.clip_len, self.image_size),
            "extrinsics_clip": default_extrinsics(self.clip_len),
            "sequence_id": self.sequence_id,
            "segment_id": seg_idx,
            "sentence": meta["sentence"],
            "frame_ids": torch.tensor(frame_ids, dtype=torch.long),
            "timestamp_ms": torch.tensor(timestamps, dtype=torch.float32),
        }


def has_both_hands(frame):
    hands = frame.get("hands", {})
    return isinstance(hands.get("left"), dict) and isinstance(hands.get("right"), dict)


def hand_to_array(hand):
    return np.asarray([hand[name] for name in JOINT_ORDER], dtype=np.float32)


def _read_rgb_video(path, image_size):
    video = read_video(str(path), pts_unit="sec", output_format="TCHW")[0].float() / 255.0
    if video.numel() == 0:
        raise RuntimeError(f"Unable to decode video: {path}")
    if video.shape[-2:] != (image_size, image_size):
        video = F.interpolate(video, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return video


def _timestamp_to_index(relative_time, num_frames):
    idx = int(round(float(relative_time) * (num_frames - 1)))
    return min(max(idx, 0), num_frames - 1)


def default_intrinsics(clip_len, image_size):
    fx = fy = float(image_size)
    cx = cy = float(image_size) / 2.0
    frame = {
        "left": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "right": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
    }
    return [frame for _ in range(clip_len)]


def default_extrinsics(clip_len):
    rot = torch.eye(3, dtype=torch.float32)
    left = {"rotation_matrix": rot, "translation": torch.zeros(1, 3, dtype=torch.float32)}
    right = {"rotation_matrix": rot, "translation": torch.tensor([[0.06, 0.0, 0.0]], dtype=torch.float32)}
    return [{"left": left, "right": right} for _ in range(clip_len)]


def default_aslhand2_roots():
    return {
        "keypoint_root": os.environ.get("ASLHAND2_KEYPOINT_ROOT", "/home/jqu11/bigdata/data/ASLHand2/hand_keypoints_synced"),
        "zed_root": os.environ.get("ASLHAND2_ZED_ROOT", "/home/jqu11/bigdata/data/ASLHand2/ZED_Segments"),
    }
