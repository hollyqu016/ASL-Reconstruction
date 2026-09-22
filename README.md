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

## Privileged front-teacher MVP

The training script supports three modes:

```bash
python main.py --train_mode baseline
python main.py --train_mode teacher --smoke
python main.py --train_mode student --teacher_ckpt checkpoints/front_teacher_best.pth
```

`baseline` preserves the original stereo EgoSSA/AIM1 behavior. `teacher` trains only the front-camera teacher and its pose head. `student` freezes a trained front teacher and adds latent distillation only when a batch contains paired front RGB and ego stereo.

Expected optional WebDataset fields per frame:

- `camera-front-{0..N}.png` / `front-{0..N}.png` / `front_camera_{0..N}.png`: synchronized front-view RGB images.
- `meta.json` optional entries: `front_camera_ids`, `front_view_confidence`, `front_intrinsics`, `front_extrinsics`.
- Optional 2D supervision for multi-camera DLT pseudo-labels: `front_right_keypoints_2d.npy`, `front_left_keypoints_2d.npy`, plus optional `front_right_keypoint_confidence.npy` and `front_left_keypoint_confidence.npy`.
- Optional precomputed 3D pseudo-labels: `front_pseudo_right_landmarks.npy`, `front_pseudo_left_landmarks.npy`.

The teacher uses a shared Swin Transformer / Swin-FPN encoder for all front cameras, camera embeddings, and confidence-aware view attention to produce a fused teacher representation. A `TeacherPoseHead` predicts synchronized left/right keypoints directly from the teacher representation:

```text
front RGB -> FrontCameraTeacher -> teacher_repr -> TeacherPoseHead -> keypoints
```

Teacher loss is supervised against real synchronized keypoints:

```text
L_teacher = L_teacher_pose + LAMBDA_VELOCITY * L_velocity
```

Student distillation is:

```text
L_distill = SmoothL1(StudentProjectionHead(student_latent), stopgrad(teacher_repr))
```

Other front-teacher environment variables: `FRONT_IMG_SIZE`, `FRONT_IN_CHANS`, `FRONT_MAX_CAMERAS`.

### ASLHand2 front teacher

Defaults:

```bash
python main.py \
  --train_mode teacher \
  --aslhand2_keypoint_root /home/jqu11/bigdata/data/ASLHand2/hand_keypoints_synced \
  --aslhand2_zed_root /home/jqu11/bigdata/data/ASLHand2/ZED_Segments \
  --aslhand2_sequence Abdul_03_52 \
  --batch_size 1 \
  --clip_len 4 \
  --smoke
```

The loader is intentionally conservative: it aligns ZED image files and keypoint files by frame id parsed from filenames, supports one or more front views, and detects whether keypoints are 2D or 3D from the actual tensor shape. It does not fabricate 3D labels from 2D keypoints and does not fabricate ego/front pairing.

Student mode requires batches that actually contain both stereo fields and front fields. If a stereo batch has no front fields, it falls back to the original AIM1 loss for that batch.

Metrics are wrist-relative, in mm, over 20 joints: MPJPE, PA-MPJPE, PCK@5, PCK@10, AUC@30.
