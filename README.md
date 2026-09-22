# EgoSSA baseline (HOT3D)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install setuptools wheel
.venv/bin/pip install --no-build-isolation chumpy==0.70
sed -i 's/inspect.getargspec(/inspect.getfullargspec(/' .venv/lib/python3.11/site-packages/chumpy/ch.py
```

MANO models (`MANO_LEFT.pkl`, `MANO_RIGHT.pkl`) are expected in `/mnt/bigdata/data/body_models/mano` (override with `MANO_PATH`).

## Data

Download the full HOT3D-Clips dataset from `bop-benchmark/hot3d` on Hugging Face, then convert:

```bash
.venv/bin/python tools/convert_hot3d_clips.py --input data/hot3d_raw/train_aria/{...}.tar --output-dir data/hot3d_wds/train
.venv/bin/python tools/convert_hot3d_clips.py --input data/hot3d_raw/train_aria/{...}.tar --output-dir data/hot3d_wds/val
.venv/bin/python tools/convert_hot3d_clips.py --input data/hot3d_raw/train_aria/{...}.tar --output-dir data/hot3d_wds/test
```

## Train / evaluate

```bash
CUDA_VISIBLE_DEVICES=0 NUM_EPOCHS=3 .venv/bin/python main.py
.venv/bin/python tools/pose.py
```

Environment variables: `DATA_TRAIN`, `DATA_VAL`, `MANO_PATH`, `LOG_DIR`, `SAVE_DIR`, `NUM_EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`, `STRIDE`, `CROP_HANDS`.

## Front-camera teacher baseline

The training script now includes an optional front-camera teacher branch. It is enabled by default (`USE_FRONT_TEACHER=1`) but only contributes losses when a batch contains front-camera fields.

Expected optional WebDataset fields per frame:

- `camera-front-{0..N}.png` / `front-{0..N}.png` / `front_camera_{0..N}.png`: synchronized front-view RGB images.
- `meta.json` optional entries: `front_camera_ids`, `front_view_confidence`, `front_intrinsics`, `front_extrinsics`.
- Optional 2D supervision for multi-camera DLT pseudo-labels: `front_right_keypoints_2d.npy`, `front_left_keypoints_2d.npy`, plus optional `front_right_keypoint_confidence.npy` and `front_left_keypoint_confidence.npy`.
- Optional precomputed 3D pseudo-labels: `front_pseudo_right_landmarks.npy`, `front_pseudo_left_landmarks.npy`.

The teacher uses a shared Swin Transformer / Swin-FPN encoder for all front cameras, camera embeddings, and confidence-aware view attention to produce a fused teacher representation. During training it adds:

- pose-level supervision from front-camera pseudo-labels, weighted by `LAMBDA_TEACHER_POSE` (default `0.5`);
- representation-level distillation from the ego student latent projection to the fused teacher representation, weighted by `LAMBDA_DISTILL` (default `0.1`).

Other front-teacher environment variables: `FRONT_IMG_SIZE`, `FRONT_IN_CHANS`, `FRONT_MAX_CAMERAS`.

Metrics are wrist-relative, in mm, over 20 joints: MPJPE, PA-MPJPE, PCK@5, PCK@10, AUC@30.
