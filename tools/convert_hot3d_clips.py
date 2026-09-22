import argparse
import io
import json
import os
import sys
import tarfile
from collections import defaultdict

import numpy as np
import torch
import webdataset as wds
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mano import create_mano_layers, mano_joints

LEFT_CAM, RIGHT_CAM = "1201-1", "1201-2"


def quat_trans_to_matrix(q_wxyz, t):
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pose_to_matrix(j):
    return quat_trans_to_matrix(j["quaternion_wxyz"], j["translation_xyz"])


def intrinsics_of(calib):
    p = calib["projection_params"]
    return {"fx": p[0], "fy": p[0], "cx": p[1], "cy": p[2]}


def read_clip(path):
    frames = defaultdict(dict)
    shapes = None
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            data = tar.extractfile(m).read()
            if m.name == "__hand_shapes.json__":
                shapes = json.loads(data)
                continue
            fid, key = m.name.split(".", 1)
            frames[int(fid)][key] = data
    return shapes, [frames[k] for k in sorted(frames)]


def to_png(jpg_bytes):
    buf = io.BytesIO()
    Image.open(io.BytesIO(jpg_bytes)).convert("L").save(buf, format="PNG")
    return buf.getvalue()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mano-dir", default="/mnt/bigdata/data/body_models/mano")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    layers = dict(zip(("left", "right"), create_mano_layers(args.mano_dir)))

    for clip_path in args.input:
        name = os.path.basename(clip_path).replace(".tar", "")
        shapes, frames = read_clip(clip_path)
        beta = torch.tensor(shapes["mano"], dtype=torch.float32)[None]
        out_path = os.path.join(args.output_dir, f"{name}.tar")
        kept = 0
        with wds.TarWriter(out_path) as sink:
            for fi, fr in enumerate(frames):
                hands = json.loads(fr["hands.json"])
                if any(hands.get(h) is None or "mano_pose" not in hands[h] for h in ("left", "right")):
                    continue
                cams = json.loads(fr["cameras.json"])
                info = json.loads(fr["info.json"])

                T_w_camL = pose_to_matrix(cams[LEFT_CAM]["T_world_from_camera"])
                T_camL_w = np.linalg.inv(T_w_camL)
                T_dev_camL = pose_to_matrix(cams[LEFT_CAM]["calibration"]["T_device_from_camera"])
                T_dev_camR = pose_to_matrix(cams[RIGHT_CAM]["calibration"]["T_device_from_camera"])

                lms = {}
                for side in ("left", "right"):
                    mp = hands[side]["mano_pose"]
                    xform = torch.tensor(mp["wrist_xform"], dtype=torch.float32)[None]
                    thetas = torch.tensor(mp["thetas"], dtype=torch.float32)[None]
                    j = mano_joints(layers[side], beta, thetas, xform[:, :3], xform[:, 3:])[0].numpy().astype(np.float64)
                    j_cam = (T_camL_w[:3, :3] @ j.T).T + T_camL_w[:3, 3]
                    lms[side] = (j_cam * 1000.0).astype(np.float32)

                extr = {}
                for side, T_dev_cam in (("left", T_dev_camL), ("right", T_dev_camR)):
                    T_cam_dev = np.linalg.inv(T_dev_cam)
                    extr[side] = {"rotation_matrix": T_cam_dev[:3, :3].tolist(), "translation": [T_cam_dev[:3, 3].tolist()]}

                boxes = {}
                for view, cam in (("left", LEFT_CAM), ("right", RIGHT_CAM)):
                    bs = [hands[s]["boxes_amodal"][cam] for s in ("left", "right")
                          if hands[s].get("boxes_amodal", {}).get(cam) is not None]
                    boxes[view] = [min(b[0] for b in bs), min(b[1] for b in bs),
                                   max(b[2] for b in bs), max(b[3] for b in bs)] if bs else None

                sink.write({
                    "__key__": f"{name}_{fi:06d}",
                    "camera-slam-left.png": to_png(fr[f"image_{LEFT_CAM}.jpg"]),
                    "camera-slam-right.png": to_png(fr[f"image_{RIGHT_CAM}.jpg"]),
                    "left_landmarks.npy": lms["left"],
                    "right_landmarks.npy": lms["right"],
                    "meta.json": {
                        "hand_boxes": boxes,
                        "intrinsics": {
                            "left": intrinsics_of(cams[LEFT_CAM]["calibration"]),
                            "right": intrinsics_of(cams[RIGHT_CAM]["calibration"]),
                        },
                        "extrinsics": extr,
                        "timestamp_ns": info["image_timestamps_ns"][LEFT_CAM],
                        "sequence_id": info["sequence_id"],
                    },
                })
                kept += 1
        print(f"{name}: kept {kept}/{len(frames)} frames -> {out_path}")


if __name__ == "__main__":
    main()
