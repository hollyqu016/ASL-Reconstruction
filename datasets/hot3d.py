import copy
from collections import deque

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms


class StructSyncClipDataset(IterableDataset):
    def __init__(self, tar_path, clip_len=8, stride=1, crop_hands=True, crop_size=224, crop_scale=1.2, min_crop=128, front_camera_keys=None):
        super().__init__()
        self.tar_path = tar_path
        self.clip_len = clip_len
        self.stride = stride
        self.crop_hands = crop_hands
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.min_crop = min_crop
        self.front_camera_keys = front_camera_keys
        self.to_tensor = transforms.ToTensor()
        self.base_dataset = wds.WebDataset(tar_path, shardshuffle=False).decode("pil")

    def _front_keys(self, sample):
        if self.front_camera_keys:
            return [k for k in self.front_camera_keys if k in sample]
        patterns = (
            "camera-front-{}.png",
            "front-{}.png",
            "front_camera_{}.png",
            "camera-front-{}.jpg",
            "front-{}.jpg",
            "front_camera_{}.jpg",
        )
        keys = []
        for idx in range(8):
            key = next((p.format(idx) for p in patterns if p.format(idx) in sample), None)
            if key is not None:
                keys.append(key)
        return keys

    def _crop(self, img, box, intr):
        W, H = img.size
        if box is None:
            x0, y0, side = 0.0, (H - W) / 2.0, float(W)
        else:
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            side = max(box[2] - box[0], box[3] - box[1], self.min_crop) * self.crop_scale
            x0, y0 = cx - side / 2.0, cy - side / 2.0
        s = self.crop_size / side
        img = img.crop((x0, y0, x0 + side, y0 + side)).resize((self.crop_size, self.crop_size), Image.BILINEAR)
        intr = {"fx": intr["fx"] * s, "fy": intr["fy"] * s, "cx": (intr["cx"] - x0) * s, "cy": (intr["cy"] - y0) * s}
        return img, intr

    def __iter__(self):
        clip_queue = deque()
        prev_clip = None
        for sample in self.base_dataset:
            key = sample["__key__"]
            left_img, right_img = sample["camera-slam-left.png"], sample["camera-slam-right.png"]
            left_lm, right_lm, meta = sample["left_landmarks.npy"], sample["right_landmarks.npy"], sample["meta.json"]
            left_img, right_img = left_img.convert("L"), right_img.convert("L")

            clip_id = key.rsplit("_", 1)[0]
            if clip_id != prev_clip:
                clip_queue.clear()
                prev_clip = clip_id

            intrinsics = copy.deepcopy(meta["intrinsics"])
            if self.crop_hands and "hand_boxes" in meta:
                left_img, intrinsics["left"] = self._crop(left_img, meta["hand_boxes"]["left"], intrinsics["left"])
                right_img, intrinsics["right"] = self._crop(right_img, meta["hand_boxes"]["right"], intrinsics["right"])

            frame = {
                "left_img": self.to_tensor(left_img),
                "right_img": self.to_tensor(right_img),
                "left_landmarks": left_lm.astype(np.float32),
                "right_landmarks": right_lm.astype(np.float32),
                "intrinsics": intrinsics,
                "extrinsics": meta["extrinsics"],
            }

            front_keys = self._front_keys(sample)
            if front_keys:
                frame["front_imgs"] = torch.stack([self.to_tensor(sample[k].convert("RGB")) for k in front_keys])
                if "front_camera_ids" in meta:
                    frame["front_camera_ids"] = np.asarray(meta["front_camera_ids"], dtype=np.int64)
                if "front_view_confidence" in meta:
                    frame["front_view_confidence"] = np.asarray(meta["front_view_confidence"], dtype=np.float32)
                if "front_intrinsics" in meta:
                    frame["front_intrinsics"] = np.asarray(meta["front_intrinsics"], dtype=np.float32)
                if "front_extrinsics" in meta:
                    frame["front_extrinsics"] = np.asarray(meta["front_extrinsics"], dtype=np.float32)
                for src, dst in (
                    ("front_right_keypoints_2d.npy", "front_right_keypoints_2d"),
                    ("front_left_keypoints_2d.npy", "front_left_keypoints_2d"),
                    ("front_right_keypoint_confidence.npy", "front_right_keypoint_confidence"),
                    ("front_left_keypoint_confidence.npy", "front_left_keypoint_confidence"),
                    ("front_pseudo_right_landmarks.npy", "front_pseudo_right_landmarks"),
                    ("front_pseudo_left_landmarks.npy", "front_pseudo_left_landmarks"),
                ):
                    if src in sample:
                        frame[dst] = sample[src].astype(np.float32)

            clip_queue.append(frame)

            if len(clip_queue) == self.clip_len:
                clip = list(clip_queue)
                out = {
                    "left_img_clip": torch.stack([f["left_img"] for f in clip]),
                    "right_img_clip": torch.stack([f["right_img"] for f in clip]),
                    "left_landmarks_clip": np.stack([f["left_landmarks"] for f in clip]),
                    "right_landmarks_clip": np.stack([f["right_landmarks"] for f in clip]),
                    "intrinsics_clip": [f["intrinsics"] for f in clip],
                    "extrinsics_clip": [f["extrinsics"] for f in clip],
                }
                if all("front_imgs" in f for f in clip):
                    out["front_img_clip"] = torch.stack([f["front_imgs"] for f in clip])
                    for name in (
                        "front_camera_ids",
                        "front_view_confidence",
                        "front_intrinsics",
                        "front_extrinsics",
                        "front_right_keypoints_2d",
                        "front_left_keypoints_2d",
                        "front_right_keypoint_confidence",
                        "front_left_keypoint_confidence",
                        "front_pseudo_right_landmarks",
                        "front_pseudo_left_landmarks",
                    ):
                        if all(name in f for f in clip):
                            out[f"{name}_clip"] = np.stack([f[name] for f in clip])
                yield out
                for _ in range(self.stride):
                    if clip_queue:
                        clip_queue.popleft()
