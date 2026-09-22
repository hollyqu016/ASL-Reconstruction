import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
KP_EXTS = {".json", ".npy", ".npz"}


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
        self.keypoint_dir = Path(keypoint_root) / sequence_id
        self.zed_dir = Path(zed_root) / sequence_id
        self.sequence_id = sequence_id
        self.clip_len = clip_len
        self.stride = stride
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()

        if not self.keypoint_dir.exists():
            raise FileNotFoundError(f"ASLHand2 keypoint directory not found: {self.keypoint_dir}")
        if not self.zed_dir.exists():
            raise FileNotFoundError(f"ASLHand2 ZED directory not found: {self.zed_dir}")

        self.image_files = self._discover_images(self.zed_dir)
        self.keypoint_files = self._discover_keypoints(self.keypoint_dir)
        self.frame_ids = sorted(set(self.image_files) & set(self.keypoint_files))
        if len(self.frame_ids) < clip_len:
            raise RuntimeError(
                f"Only found {len(self.frame_ids)} synchronized ASLHand2 frames for {sequence_id}; "
                f"need at least clip_len={clip_len}."
            )
        self.starts = list(range(0, len(self.frame_ids) - clip_len + 1, stride))

        sample_kp = self._read_keypoints(self.keypoint_files[self.frame_ids[0]])[0]
        self.keypoint_dim = sample_kp.shape[-1]
        print(
            f"ASLHand2 {sequence_id}: {len(self.frame_ids)} synced frames, "
            f"{self.num_views} front view(s), keypoints are {self.keypoint_dim}D."
        )

    @staticmethod
    def _frame_id(path):
        digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
        return digits[-1] if digits else path.stem

    def _discover_images(self, root):
        by_frame = {}
        for path in root.rglob("*"):
            if path.suffix.lower() not in IMG_EXTS:
                continue
            frame_id = self._frame_id(path)
            by_frame.setdefault(frame_id, []).append(path)
        if not by_frame:
            raise RuntimeError(f"No image files found under {root}")
        self.num_views = max(len(v) for v in by_frame.values())
        return {k: sorted(v) for k, v in by_frame.items()}

    def _discover_keypoints(self, root):
        found = {}
        for path in root.rglob("*"):
            if path.suffix.lower() not in KP_EXTS:
                continue
            found[self._frame_id(path)] = path
        if not found:
            raise RuntimeError(f"No keypoint files found under {root}")
        return found

    def _read_image(self, path):
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        return self.to_tensor(img)

    def _read_keypoints(self, path):
        if path.suffix.lower() == ".npy":
            data = np.load(path, allow_pickle=True)
        elif path.suffix.lower() == ".npz":
            data = dict(np.load(path, allow_pickle=True))
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return normalize_keypoints(data)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        ids = self.frame_ids[self.starts[idx]: self.starts[idx] + self.clip_len]
        images, valid, left_kps, right_kps, left_conf, right_conf = [], [], [], [], [], []
        for frame_id in ids:
            frame_imgs = [self._read_image(p) for p in self.image_files[frame_id]]
            frame_valid = [True] * len(frame_imgs)
            while len(frame_imgs) < self.num_views:
                frame_imgs.append(torch.zeros_like(frame_imgs[0]))
                frame_valid.append(False)
            kp, conf = self._read_keypoints(self.keypoint_files[frame_id])
            left_kps.append(torch.from_numpy(kp["left"]).float())
            right_kps.append(torch.from_numpy(kp["right"]).float())
            left_conf.append(torch.from_numpy(conf["left"]).float())
            right_conf.append(torch.from_numpy(conf["right"]).float())
            images.append(torch.stack(frame_imgs))
            valid.append(torch.tensor(frame_valid, dtype=torch.bool))

        return {
            "front_rgb": torch.stack(images),
            "front_valid": torch.stack(valid),
            "front_left_keypoints": torch.stack(left_kps),
            "front_right_keypoints": torch.stack(right_kps),
            "front_left_confidence": torch.stack(left_conf),
            "front_right_confidence": torch.stack(right_conf),
            "sequence_id": self.sequence_id,
            "frame_ids": ids,
        }


def normalize_keypoints(data):
    if isinstance(data, np.ndarray):
        arr = data.item() if data.dtype == object and data.shape == () else data
    else:
        arr = data

    if isinstance(arr, dict):
        left = first_present(arr, ("left", "left_hand", "left_keypoints", "hand_left"))
        right = first_present(arr, ("right", "right_hand", "right_keypoints", "hand_right"))
        if left is None or right is None:
            keypoints = first_present(arr, ("keypoints", "joints", "points"))
            if keypoints is not None:
                keypoints = np.asarray(keypoints, dtype=np.float32)
                if keypoints.shape[-3] == 2:
                    left, right = keypoints[0], keypoints[1]
        left, left_conf = split_confidence(left)
        right, right_conf = split_confidence(right)
    else:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape[-3] != 2:
            raise ValueError(f"Cannot infer left/right hand layout from keypoint shape {arr.shape}")
        left, left_conf = split_confidence(arr[0])
        right, right_conf = split_confidence(arr[1])

    if left is None or right is None:
        raise ValueError("Could not find both left and right hand keypoints")
    return {"left": left, "right": right}, {"left": left_conf, "right": right_conf}


def first_present(mapping, names):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def split_confidence(value):
    if isinstance(value, dict):
        coords = first_present(value, ("keypoints", "joints", "points", "coords"))
        conf = first_present(value, ("confidence", "conf", "scores", "score"))
        coords = np.asarray(coords, dtype=np.float32)
        if conf is None:
            conf = np.ones(coords.shape[:-1], dtype=np.float32)
        return coords, np.asarray(conf, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[-1] >= 4:
        return arr[..., :3], arr[..., 3]
    if arr.shape[-1] == 3:
        coords = arr
        conf = np.ones(arr.shape[:-1], dtype=np.float32)
        return coords, conf
    if arr.shape[-1] == 2:
        coords = arr
        conf = np.ones(arr.shape[:-1], dtype=np.float32)
        return coords, conf
    raise ValueError(f"Unsupported keypoint shape {arr.shape}")


def default_aslhand2_roots():
    return {
        "keypoint_root": os.environ.get("ASLHAND2_KEYPOINT_ROOT", "/home/jqu11/bigdata/data/ASLHand2/hand_keypoints_synced"),
        "zed_root": os.environ.get("ASLHAND2_ZED_ROOT", "/home/jqu11/bigdata/data/ASLHand2/ZED_Segments"),
    }
