import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_video


class ASLRepairFrontVideoDataset(Dataset):
    """Frontal ASL Repair videos with clip-level semantic labels.

    This dataset is intentionally unpaired with ASLHand2. It returns front_video
    clips and metadata for clip-level semantic/gesture objectives only.
    """

    def __init__(self, root, clip_len=16, image_size=224, split=None, participants=None, split_ratios=(0.7, 0.15, 0.15), max_samples=None):
        self.root = Path(root)
        self.clip_len = clip_len
        self.image_size = image_size
        if not self.root.exists():
            raise FileNotFoundError(f"ASL Repair root not found: {self.root}")

        self.item_metadata = self._load_items()
        all_samples = self._load_manifest()
        all_item_ids = sorted({s["item_id"] for s in all_samples if s.get("item_id")})
        self.item_to_idx = {item_id: i for i, item_id in enumerate(all_item_ids)}
        selected_participants = _select_participants([s["participant_id"] for s in all_samples], participants, split, split_ratios)
        self.participants = selected_participants
        self.samples = [s for s in all_samples if s["participant_id"] in set(selected_participants)]
        if max_samples:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError(f"No ASL Repair videos found under {self.root}")

        print(f"ASL Repair {split or 'custom'}: {len(self.samples)} clips, {len(self.item_to_idx)} item classes, participants={self.participants}.")

    def _load_items(self):
        items = {}
        path = self.root / "items.csv"
        if not path.exists():
            return items
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                item_id = row.get("item_id") or row.get("id")
                if item_id:
                    items[item_id] = row
        return items

    def _load_manifest(self):
        manifest = self.root / "manifest.csv"
        if manifest.exists():
            rows = []
            with open(manifest, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    video_rel = _first_present(row, ("path", "video_path", "filepath", "file"))
                    if not video_rel:
                        continue
                    path = self.root / video_rel
                    if path.exists():
                        rows.append(self._normalize_row(row, path))
            return rows
        return self._scan_videos()

    def _scan_videos(self):
        rows = []
        for path in sorted(self.root.glob("V*/**/*.webm")):
            parts = path.relative_to(self.root).parts
            participant = parts[0] if parts else ""
            item_id = parts[1] if len(parts) > 1 else ""
            condition = path.stem if path.parent.name != "repair" else f"repair/{path.stem}"
            rows.append(self._normalize_row({"participant_id": participant, "item_id": item_id, "condition": condition}, path))
        return rows

    def _normalize_row(self, row, path):
        item_id = row.get("item_id") or row.get("message_id") or row.get("stimulus_id") or ""
        participant_id = row.get("participant_id") or row.get("participant") or path.parts[-3]
        condition = row.get("condition") or row.get("clip_type") or path.stem
        meta = self.item_metadata.get(item_id, {})
        return {
            "path": path,
            "participant_id": participant_id,
            "item_id": item_id,
            "condition": condition,
            "reference_english": _first_present(row, ("clip_reference_english", "full_reference_english", "reference_english", "english", "message")) or _first_present(meta, ("full_reference_english", "corpus_english", "reference_english", "english", "message", "sentence")) or "",
            "reference_gloss": _first_present(row, ("segment_reference_gloss", "intended_gloss", "intended_reference_gloss", "reference_gloss", "gloss")) or _first_present(meta, ("intended_gloss", "intended_reference_gloss", "reference_gloss", "gloss")) or "",
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video = _read_clip(sample["path"], self.clip_len, self.image_size)
        item_id = sample["item_id"]
        return {
            "front_video": video,
            "item_label": torch.tensor(self.item_to_idx.get(item_id, -1), dtype=torch.long),
            "participant_id": sample["participant_id"],
            "item_id": item_id,
            "condition": sample["condition"],
            "reference_english": sample["reference_english"],
            "reference_gloss": sample["reference_gloss"],
            "video_path": str(sample["path"]),
        }


def _first_present(row, names):
    for name in names:
        value = row.get(name)
        if value:
            return value
    return ""


def _select_participants(participant_values, participants=None, split=None, split_ratios=(0.7, 0.15, 0.15)):
    all_participants = sorted({p for p in participant_values if p})
    if participants:
        requested = [p.strip() for p in str(participants).split(",") if p.strip()]
        all_participants = [p for p in all_participants if p in set(requested)]
    if not split:
        return all_participants
    n = len(all_participants)
    n_train = max(1, int(round(n * split_ratios[0]))) if n else 0
    n_val = max(1, int(round(n * split_ratios[1]))) if n >= 3 else max(0, n - n_train)
    if n_train + n_val >= n and n > 1:
        n_val = 1
        n_train = max(1, n - 2)
    splits = {
        "train": all_participants[:n_train],
        "val": all_participants[n_train:n_train + n_val],
        "test": all_participants[n_train + n_val:],
    }
    return splits[split]


def _read_clip(path, clip_len, image_size):
    video, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
    if video.numel() == 0:
        raise RuntimeError(f"Unable to decode video: {path}")
    video = video.float() / 255.0
    idx = torch.linspace(0, video.shape[0] - 1, steps=clip_len).round().long()
    video = video[idx]
    if video.shape[-2:] != (image_size, image_size):
        video = F.interpolate(video, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return video


def load_ground_truth(root):
    path = Path(root) / "ground_truth.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_asl_repair_root():
    return os.environ.get("ASL_REPAIR_ROOT", "/home/jqu11/bigdata/data/ASL_Repair_Videos_2026-09-08")
