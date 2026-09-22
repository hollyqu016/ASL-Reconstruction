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


class ASLHand2FrontDataset(Dataset):
    def __init__(
        self,
        keypoint_root,
        zed_root,
        sequence_id="Abdul_03_52",
        clip_len=8,
        stride=1,
        image_size=224,
    ):
        self.keypoint_dir = Path(keypoint_root) / sequence_id / "keypoints_label"
        self.zed_dir = Path(zed_root) / sequence_id
        self.sequence_id = sequence_id
        self.clip_len = clip_len
        self.stride = stride
        self.image_size = image_size
        self.keypoint_dim = 3
        self.num_views = 2

        if not self.keypoint_dir.exists():
            raise FileNotFoundError(f"ASLHand2 keypoint directory not found: {self.keypoint_dir}")
        if not self.zed_dir.exists():
            raise FileNotFoundError(f"ASLHand2 ZED directory not found: {self.zed_dir}")

        self.segments = self._discover_segments()
        self.windows = []
        for seg_idx, meta in self.segments.items():
            n = len(meta["frames"])
            self.windows.extend((seg_idx, s) for s in range(0, n - clip_len + 1, stride))
        if not self.windows:
            raise RuntimeError(f"No ASLHand2 windows found for {sequence_id} with clip_len={clip_len}.")

        print(
            f"ASLHand2 {sequence_id}: {len(self.segments)} synced segments, "
            f"{len(self.windows)} windows, ZED left/right front views, 21 joints, 3D keypoints."
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
            frames = [fr for fr in meta["frames"] if has_both_hands(fr)]
            if len(frames) < self.clip_len:
                continue
            segments[seg_idx] = {
                "keypoint_path": kp_path,
                "left_video": left_video,
                "right_video": right_video,
                "frames": frames,
                "start_timestamp_ms": float(meta["start_timestamp_ms"]),
                "end_timestamp_ms": float(meta["end_timestamp_ms"]),
                "sentence": meta.get("sentence", ""),
            }
        if not segments:
            raise RuntimeError(f"No synchronized ASLHand2 segment triplets found under {self.keypoint_dir} and {self.zed_dir}")
        return segments

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seg_idx, start = self.windows[idx]
        meta = self.segments[seg_idx]
        frames = meta["frames"][start:start + self.clip_len]

        left_video = read_video(str(meta["left_video"]), pts_unit="sec", output_format="TCHW")[0].float() / 255.0
        right_video = read_video(str(meta["right_video"]), pts_unit="sec", output_format="TCHW")[0].float() / 255.0
        videos = [resize_video(left_video, self.image_size), resize_video(right_video, self.image_size)]

        duration = max(meta["end_timestamp_ms"] - meta["start_timestamp_ms"], 1e-6)
        image_frames, left_kps, right_kps, frame_ids = [], [], [], []
        for fr in frames:
            rel = (float(fr["timestamp_ms"]) - meta["start_timestamp_ms"]) / duration
            view_imgs = []
            for video in videos:
                vid_idx = int(round(rel * (video.shape[0] - 1)))
                vid_idx = min(max(vid_idx, 0), video.shape[0] - 1)
                view_imgs.append(video[vid_idx])
            image_frames.append(torch.stack(view_imgs))
            left_kps.append(torch.tensor(hand_to_array(fr["hands"]["left"]), dtype=torch.float32))
            right_kps.append(torch.tensor(hand_to_array(fr["hands"]["right"]), dtype=torch.float32))
            frame_ids.append(int(fr["frame"]))

        return {
            "front_rgb": torch.stack(image_frames),
            "front_valid": torch.ones(self.clip_len, self.num_views, dtype=torch.bool),
            "front_left_keypoints": torch.stack(left_kps),
            "front_right_keypoints": torch.stack(right_kps),
            "front_left_confidence": torch.ones(self.clip_len, len(JOINT_ORDER), dtype=torch.float32),
            "front_right_confidence": torch.ones(self.clip_len, len(JOINT_ORDER), dtype=torch.float32),
            "sequence_id": self.sequence_id,
            "segment_id": seg_idx,
            "sentence": meta["sentence"],
            "frame_ids": frame_ids,
        }


def has_both_hands(frame):
    hands = frame.get("hands", {})
    return isinstance(hands.get("left"), dict) and isinstance(hands.get("right"), dict)


def hand_to_array(hand):
    return np.asarray([hand[name] for name in JOINT_ORDER], dtype=np.float32)


def resize_video(video, image_size):
    if video.shape[-2:] == (image_size, image_size):
        return video
    return F.interpolate(video, size=(image_size, image_size), mode="bilinear", align_corners=False)


def default_aslhand2_roots():
    return {
        "keypoint_root": os.environ.get("ASLHAND2_KEYPOINT_ROOT", "/home/jqu11/bigdata/data/ASLHand2/hand_keypoints_synced"),
        "zed_root": os.environ.get("ASLHAND2_ZED_ROOT", "/home/jqu11/bigdata/data/ASLHand2/ZED_Segments"),
    }
