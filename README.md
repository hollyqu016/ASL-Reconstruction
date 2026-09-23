# EgoSSA / AIM2 unpaired baseline

This repository now keeps the original ego-stereo reconstruction path and adds a first AIM2 baseline that matches the actual datasets.

## Dataset roles

- **ASLHand2**: egocentric ZED stereo. `ZED_Segments/<sequence>/left` and `right` are the two ego cameras. `hand_keypoints_synced/<sequence>/keypoints_label/segment_*.json` provides synchronized 3D hand keypoints for those segments.
- **ASL Repair**: independent frontal webcam repair videos. `manifest.csv`, `items.csv`, and `ground_truth.json` describe clip-level references, conditions, and message/gloss prompts. They are not frame-level pose ground truth.
- **HOT3D**: remains supported by `--train_mode baseline`, but the inspected HOT3D data only contains head-mounted egocentric views, not true external front cameras.

The implementation does **not** perform frame-level front-to-ego distillation, because ASLHand2 and ASL Repair are not synchronized recordings of the same performance.

## AIM2 V1 training modes

```bash
python main.py --train_mode pose
python main.py --train_mode ego --pose_ckpt checkpoints/pose_autoencoder_best.pth
python main.py --train_mode front
python main.py --train_mode joint_unpaired --pose_ckpt checkpoints/pose_autoencoder_best.pth
python main.py --train_mode baseline
```

### `pose`

Trains a pose autoencoder on ASLHand2 3D joints:

```text
ASLHand2 3D joints -> PoseEncoder -> z_pose [B,T,256] -> PoseDecoder -> reconstructed joints
```

Losses:

- SmoothL1 joint reconstruction.
- Optional temporal velocity reconstruction.

### `ego`

Trains ego-stereo reconstruction on ASLHand2:

```text
ASLHand2 ego left/right -> AIM1/EgoSSA -> z_ego -> MANO decoder -> 3D hands
```

If a pose autoencoder checkpoint exists, this also aligns:

```text
StudentProjection(z_ego) -> stopgrad(PoseEncoder(GT joints))
```

This alignment is valid because both terms come from the same ASLHand2 synchronized ego/keypoint sample.

### `front`

Trains the unpaired ASL Repair frontal branch:

```text
ASL Repair front video -> FrontVideoEncoder -> z_front_clip [B,256] -> item_id classifier
```

This is a clip-level semantic/gesture objective. It does not use frame-level gloss or pose labels.

### `joint_unpaired`

Alternates ASLHand2 ego batches and ASL Repair front batches in the same optimization loop:

- ASLHand2: 3D pose supervision plus optional pose-latent alignment.
- ASL Repair: clip-level `item_id` classification.

It does not zip arbitrary ASLHand2 and ASL Repair samples as if they were paired.

## Dataset paths on dragon

```bash
python main.py --train_mode pose \
  --aslhand2_keypoint_root /home/jqu11/bigdata/data/ASLHand2/hand_keypoints_synced \
  --aslhand2_zed_root /home/jqu11/bigdata/data/ASLHand2/ZED_Segments \
  --aslhand2_sequence Abdul_03_52 \
  --batch_size 1 --clip_len 4 --smoke

python main.py --train_mode front \
  --asl_repair_root /home/jqu11/bigdata/data/ASL_Repair_Videos_2026-09-08 \
  --batch_size 1 --front_clip_len 8 --smoke
```

## Files added

- `datasets/aslhand2.py`: ASLHand2 ego-stereo/keypoint loader.
- `datasets/asl_repair.py`: ASL Repair frontal video/metadata loader.
- `models/privileged/pose_encoder.py`: pose encoder/decoder autoencoder.
- `models/privileged/front_video_encoder.py`: front clip encoder.
- `models/privileged/shared_projector.py`: ego-to-pose projection and shared projection utilities.

## Current limitations

- ASLHand2 camera calibration was not found in the inspected segment JSON/video triplets, so the ASLHand2 loader emits a conservative placeholder stereo calibration for EgoSSA's epipolar module. Replace this with real ZED calibration if available.
- Exact normalized label overlap between inspected ASLHand2 segment `sentence` fields and ASL Repair `item_id`/English/gloss/reference fields was 0, so prototype-level cross-dataset class alignment is disabled in V1.
- ASL Repair `ground_truth.json` is treated as reference metadata, not pose or frame-level annotation.
