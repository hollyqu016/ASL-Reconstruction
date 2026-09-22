import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.hot3d import StructSyncClipDataset
from evaluator import PoseEvaluator
from main import clip_len, glob_tars, to_relative


def frames(path):
    ds = StructSyncClipDataset(glob_tars(path), clip_len=clip_len, stride=clip_len, crop_hands=False)
    for s in ds:
        yield (to_relative(torch.from_numpy(s["right_landmarks_clip"]) / 1000.0),
               to_relative(torch.from_numpy(s["left_landmarks_clip"]) / 1000.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/hot3d_wds/train")
    ap.add_argument("--val", default="data/hot3d_wds/val")
    args = ap.parse_args()

    tr = list(frames(args.train))
    mean_r = torch.cat([r for r, _ in tr]).mean(0)
    mean_l = torch.cat([l for _, l in tr]).mean(0)

    ev = PoseEvaluator()
    for r, l in frames(args.val):
        ev.update(mean_r.expand_as(r), mean_l.expand_as(l), r, l)
    res = ev.compute()
    print(f"EgoSSA ({res['num_frames']} frames): MPJPE {res['MPJPE']:.2f} | "
          f"PA-MPJPE {res['PA-MPJPE']:.2f} | PCK@5 {res['PCK@5']:.2f} | PCK@10 {res['PCK@10']:.2f} | AUC@30 {res['AUC@30']:.2f}")


if __name__ == "__main__":
    main()
