import argparse
import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.asl_repair import ASLRepairFrontVideoDataset, default_asl_repair_root
from datasets.aslhand2 import ASLHand2EgoStereoDataset, default_aslhand2_roots, discover_aslhand2_sequences, participant_split
from datasets.hot3d import StructSyncClipDataset
from evaluator import PoseEvaluator
from loss import PoseAutoencoderLoss, StructureLatentLoss
from models.DualSwinFeatureExtractor import DualSwinFPN
from models.LatentProcessingModule import LatentProcessingModule
from models.MANODecoder import Decoder
from models.MultiScaleCrossViewFusion import MultiScaleCrossViewFusion
from models.mano import WRIST_IDX
from models.privileged import FrontVideoEncoder, PoseAutoencoder, PoseEncoder, StudentProjectionHead
from models.privileged.pose_encoder import POSE_REP_DIM, make_bimanual_pose_representation


data_tar_train = os.environ.get("DATA_TRAIN", "data/hot3d_wds/train")
data_tar_val = os.environ.get("DATA_VAL", "data/hot3d_wds/val")
mano_path = os.environ.get("MANO_PATH", "/mnt/bigdata/data/body_models/mano")
log_dir = os.environ.get("LOG_DIR", "runs/EgoSSA")
save_dir = os.environ.get("SAVE_DIR", "checkpoints")
clip_len = int(os.environ.get("CLIP_LEN", 8))
stride = int(os.environ.get("STRIDE", 1))
batch_size = int(os.environ.get("BATCH_SIZE", 4))
num_workers = int(os.environ.get("NUM_WORKERS", 4))
num_epochs = int(os.environ.get("NUM_EPOCHS", 3))
crop_hands = os.environ.get("CROP_HANDS", "1") == "1"
lr = float(os.environ.get("LR", 1e-4))
lambda_pose_align = float(os.environ.get("LAMBDA_POSE_ALIGN", 0.1))
lambda_front_cls = float(os.environ.get("LAMBDA_FRONT_CLS", 1.0))
kl_max = 0.01
kl_anneal_steps = 5000
tau_anneal_steps = 10000
lambda_gate = 1e-3
img_size = int(os.environ.get("IMG_SIZE", 224))
seed = 42
IMG_MEAN, IMG_STD = 0.449, 0.226


def set_seed(value):
    torch.manual_seed(value)
    np.random.seed(value)
    random.seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def to_relative(joints):
    return joints - joints[..., WRIST_IDX:WRIST_IDX + 1, :]


def glob_tars(path):
    return sorted(glob.glob(os.path.join(path, "*.tar"))) if os.path.isdir(path) else path


def preprocess_images(x):
    B, T, C, H, W = x.shape
    x = x.view(B * T, C, H, W)
    if (H, W) != (img_size, img_size):
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    if C == 3:
        x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    x = (x - IMG_MEAN) / IMG_STD
    return x.view(B, T, 1, img_size, img_size)


def preprocess_front_video(x):
    B, T, C, H, W = x.shape
    x = x.view(B * T, C, H, W)
    if (H, W) != (img_size, img_size):
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x.view(B, T, C, img_size, img_size)


def linear_schedule(step, start, end, num_steps):
    return end if step >= num_steps else start + (end - start) * step / num_steps


def build_camera_matrices(batch, device):
    if "intrinsics_clip" not in batch or "extrinsics_clip" not in batch:
        return None, None, None, None
    intrinsics_clip, extrinsics_clip = batch["intrinsics_clip"], batch["extrinsics_clip"]
    T, B = len(intrinsics_clip), intrinsics_clip[0]["left"]["fx"].shape[0]
    K_left = torch.zeros(T, B, 3, 3, dtype=torch.float32, device=device)
    K_right = torch.zeros(T, B, 3, 3, dtype=torch.float32, device=device)
    E_left = torch.zeros(T, B, 4, 4, dtype=torch.float32, device=device)
    E_right = torch.zeros(T, B, 4, 4, dtype=torch.float32, device=device)
    for t in range(T):
        for K, intr in ((K_left, intrinsics_clip[t]["left"]), (K_right, intrinsics_clip[t]["right"])):
            K[t, :, 0, 0], K[t, :, 1, 1] = intr["fx"].float(), intr["fy"].float()
            K[t, :, 0, 2], K[t, :, 1, 2] = intr["cx"].float(), intr["cy"].float()
            K[t, :, 2, 2] = 1.0
        for E, extr in ((E_left, extrinsics_clip[t]["left"]), (E_right, extrinsics_clip[t]["right"])):
            rot = _batched_rotation(extr["rotation_matrix"], B, device)
            trans = _batched_translation(extr["translation"], B, device)
            E[t, :, :3, :3] = rot
            E[t, :, :3, 3] = trans
            E[t, :, 3, 3] = 1.0
    K_left_inv = torch.linalg.inv(K_left)
    T_left2right = E_right @ torch.linalg.inv(E_left)
    return K_left.transpose(0, 1), K_right.transpose(0, 1), K_left_inv.transpose(0, 1), T_left2right.transpose(0, 1)


def _batched_rotation(rot, batch_size, device):
    if torch.is_tensor(rot):
        rot = rot.float().to(device)
        if rot.dim() == 3:
            return rot
        if rot.dim() == 2:
            return rot.unsqueeze(0).expand(batch_size, -1, -1)
    out = torch.zeros(batch_size, 3, 3, dtype=torch.float32, device=device)
    for r in range(3):
        for c in range(3):
            value = rot[r][c]
            out[:, r, c] = value.float().to(device) if torch.is_tensor(value) else float(value)
    return out


def _batched_translation(trans, batch_size, device):
    if torch.is_tensor(trans):
        trans = trans.float().to(device)
        if trans.dim() == 3:
            return trans[:, 0, :]
        if trans.dim() == 2:
            return trans if trans.shape[0] == batch_size else trans[0].unsqueeze(0).expand(batch_size, -1)
        if trans.dim() == 1:
            return trans.unsqueeze(0).expand(batch_size, -1)
    if isinstance(trans, (list, tuple)) and len(trans) == 1 and isinstance(trans[0], (list, tuple, torch.Tensor)):
        trans = trans[0]
    out = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
    for r in range(3):
        value = trans[r]
        out[:, r] = value.float().to(device) if torch.is_tensor(value) else float(value)
    return out


def stack_pose_gt(batch, device):
    left = batch["left_landmarks_clip"].to(device).float() / 1000.0
    right = batch["right_landmarks_clip"].to(device).float() / 1000.0
    return make_bimanual_pose_representation(left, right)


def forward_ego(batch, device, dual_swin, fusion, latent, decoder):
    left, right = batch["left_img_clip"].to(device), batch["right_img_clip"].to(device)
    K_left, _, K_left_inv, T_left2right = build_camera_matrices(batch, device)
    lf, rf = dual_swin(preprocess_images(left), preprocess_images(right))
    fused_feats = fusion(lf, rf, K_left, K_left_inv, T_left2right)
    pooled_feats = []
    for f in fused_feats:
        B, T, C, H, W = f.shape
        pooled_feats.append(F.adaptive_avg_pool2d(f.reshape(B * T, C, H, W), (1, 1)).view(B, T, C))
    z_r, z_l, aux = latent(pooled_feats)
    j_l, j_r = decoder(z_l, z_r)
    r_gt = to_relative(batch["right_landmarks_clip"].to(device).float() / 1000.0)
    l_gt = to_relative(batch["left_landmarks_clip"].to(device).float() / 1000.0)
    return to_relative(j_r), to_relative(j_l), r_gt, l_gt, aux


def unwrap(module):
    return module.module if hasattr(module, "module") else module


def maybe_parallel(module, device):
    return torch.nn.DataParallel(module.to(device)) if device.type == "cuda" else module.to(device)


def parse_args():
    roots = default_aslhand2_roots()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mode", choices=("baseline", "pose", "ego", "front", "joint_unpaired"), default=os.environ.get("TRAIN_MODE", "ego"))
    parser.add_argument("--data_train", default=data_tar_train)
    parser.add_argument("--data_val", default=data_tar_val)
    parser.add_argument("--aslhand2_keypoint_root", default=roots["keypoint_root"])
    parser.add_argument("--aslhand2_zed_root", default=roots["zed_root"])
    parser.add_argument("--aslhand2_sequence", default=os.environ.get("ASLHAND2_SEQUENCE", "Abdul_03_52"), help="Single sequence for smoke/debug compatibility.")
    parser.add_argument("--aslhand2_sequences", default=os.environ.get("ASLHAND2_SEQUENCES", ""), help="Comma-separated explicit sequence list. Empty discovers all.")
    parser.add_argument("--aslhand2_participants", default=os.environ.get("ASLHAND2_PARTICIPANTS", ""), help="Comma-separated participant allowlist.")
    parser.add_argument("--max_timestamp_mismatch_ms", type=float, default=float(os.environ.get("MAX_TIMESTAMP_MISMATCH_MS", 50.0)))
    parser.add_argument("--use_epipolar_geometry", default=os.environ.get("USE_EPIPOLAR_GEOMETRY", "false"))
    parser.add_argument("--asl_repair_root", default=default_asl_repair_root())
    parser.add_argument("--asl_repair_participants", default=os.environ.get("ASL_REPAIR_PARTICIPANTS", ""))
    parser.add_argument("--pose_ckpt", default=os.path.join(save_dir, "pose_autoencoder_best.pth"))
    parser.add_argument("--num_epochs", type=int, default=num_epochs)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--num_workers", type=int, default=num_workers)
    parser.add_argument("--clip_len", type=int, default=clip_len)
    parser.add_argument("--front_clip_len", type=int, default=int(os.environ.get("FRONT_CLIP_LEN", 16)))
    parser.add_argument("--lambda_pose_align", type=float, default=lambda_pose_align)
    parser.add_argument("--lambda_front_cls", type=float, default=lambda_front_cls)
    parser.add_argument("--freeze_front_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained_front_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def str_to_bool(value):
    return str(value).lower() in {"1", "true", "yes", "y"}


def find_calibration_candidates(*roots):
    keywords = ("calib", "camera", "intrinsic", "extrinsic", "zed")
    matches = []
    for root in roots:
        root_path = Path(root)
        if root_path.is_file():
            root_path = root_path.parent
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if path.is_file() and any(k in path.name.lower() for k in keywords):
                matches.append(str(path))
                if len(matches) >= 20:
                    return matches
    return matches


def report_calibration_state(args):
    candidates = find_calibration_candidates(args.aslhand2_keypoint_root, args.aslhand2_zed_root)
    if candidates:
        print("Potential ASLHand2/ZED calibration files found:")
        for path in candidates[:20]:
            print(f"  {path}")
    else:
        print("No ASLHand2/ZED calibration files found under the configured roots.")
    if str_to_bool(args.use_epipolar_geometry) and not candidates:
        raise RuntimeError("Epipolar geometry was requested, but no real calibration file was found. Rerun with --use_epipolar_geometry false.")
    return candidates


def make_aslhand2_loader(args, split="train", max_windows=None, shuffle=True):
    sequence_ids = args.aslhand2_sequences
    if args.smoke and not sequence_ids:
        sequence_ids = args.aslhand2_sequence
    ds = ASLHand2EgoStereoDataset(
        args.aslhand2_keypoint_root,
        args.aslhand2_zed_root,
        sequence_ids=sequence_ids,
        participants=args.aslhand2_participants,
        split=None if args.smoke else split,
        clip_len=args.clip_len,
        stride=stride,
        image_size=img_size,
        max_windows=max_windows,
        max_timestamp_mismatch_ms=args.max_timestamp_mismatch_ms,
    )
    return DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=shuffle), ds


def make_repair_loader(args, split="train", max_samples=None, shuffle=True):
    ds = ASLRepairFrontVideoDataset(
        args.asl_repair_root,
        clip_len=args.front_clip_len,
        image_size=img_size,
        split=None if args.smoke else split,
        participants=args.asl_repair_participants,
        max_samples=max_samples,
    )
    return DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=shuffle), ds


def train_pose_epoch(loader, device, pose_ae, criterion, optimizer, writer, global_step):
    pose_ae.train()
    losses = []
    for batch in tqdm(loader, desc="Pose AE", dynamic_ncols=True, mininterval=30):
        joints = stack_pose_gt(batch, device)
        optimizer.zero_grad()
        out = pose_ae(joints)
        loss, stats = criterion(out["pose_recon"], joints)
        loss.backward()
        optimizer.step()
        if writer:
            writer.add_scalar("PoseAE/total", loss.item(), global_step)
            writer.add_scalar("PoseAE/recon", stats["pose_recon"], global_step)
            writer.add_scalar("PoseAE/velocity", stats["pose_velocity"], global_step)
        losses.append(loss.item())
        global_step += 1
    return float(np.mean(losses)), global_step


def train_ego_epoch(loader, device, modules, pose_encoder, optimizer, writer, global_step, evaluator, use_pose_align=True):
    dual_swin, fusion, latent, decoder, student_proj = modules
    for m in modules:
        m.train()
    if pose_encoder is not None:
        pose_encoder.eval()
        pose_encoder.requires_grad_(False)
    loss_module = StructureLatentLoss(lambda_dyn=1.0, lambda_gate=lambda_gate)
    mpjpe = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="Ego", dynamic_ncols=True, mininterval=30)):
        lambda_kl = linear_schedule(global_step, 0.0, kl_max, kl_anneal_steps)
        unwrap(latent).tau = linear_schedule(global_step, 2.0, 0.5, tau_anneal_steps)
        optimizer.zero_grad()
        j_r, j_l, r_gt, l_gt, aux = forward_ego(batch, device, dual_swin, fusion, latent, decoder)
        pose_loss = ((j_r - r_gt).norm(dim=-1).mean() + (j_l - l_gt).norm(dim=-1).mean()) * 500.0
        latent_loss, loss_dict = loss_module(aux, lambda_kl)
        align_loss = j_r.new_tensor(0.0)
        if use_pose_align and pose_encoder is not None:
            with torch.no_grad():
                z_pose_gt = pose_encoder(stack_pose_gt(batch, device))
            align_loss = F.smooth_l1_loss(student_proj(aux), z_pose_gt.detach())
        total_loss = pose_loss + latent_loss + lambda_pose_align * align_loss
        total_loss.backward()
        optimizer.step()

        metrics = evaluator.evaluate(j_r, j_l, r_gt, l_gt)
        mpjpe.append((metrics["MPJPE_R"] + metrics["MPJPE_L"]) / 2.0)
        if batch_idx == 0:
            print(f"First ego batch: pred R/L {tuple(j_r.shape)} / {tuple(j_l.shape)}, pose-align {align_loss.item():.6f}")
        if writer:
            writer.add_scalar("Ego/total", total_loss.item(), global_step)
            writer.add_scalar("Ego/pose", pose_loss.item(), global_step)
            writer.add_scalar("Ego/pose_align", align_loss.item(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"Ego/{k}", v, global_step)
        global_step += 1
    return float(np.mean(mpjpe)), global_step


def train_front_epoch(loader, device, front_encoder, classifier, optimizer, writer, global_step):
    front_encoder.train()
    classifier.train()
    losses, accs = [], []
    for batch_idx, batch in enumerate(tqdm(loader, desc="Front", dynamic_ncols=True, mininterval=30)):
        video = preprocess_front_video(batch["front_video"].to(device).float())
        labels = batch["item_label"].to(device)
        valid = labels >= 0
        if not valid.any():
            continue
        optimizer.zero_grad()
        out = front_encoder(video)
        logits = classifier(out["z_front_clip"])
        loss = F.cross_entropy(logits[valid], labels[valid]) * lambda_front_cls
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=-1)[valid] == labels[valid]).float().mean().item()
        if batch_idx == 0:
            print(f"First front batch: video {tuple(video.shape)}, z_front_clip {tuple(out['z_front_clip'].shape)}, classes {classifier.out_features}")
        if writer:
            writer.add_scalar("Front/cls_loss", loss.item(), global_step)
            writer.add_scalar("Front/acc", acc, global_step)
        losses.append(loss.item())
        accs.append(acc)
        global_step += 1
    return float(np.mean(losses)), float(np.mean(accs)), global_step


@torch.no_grad()
def run_front_validation_batch(loader, device, front_encoder, classifier):
    front_encoder.eval()
    classifier.eval()
    batch = next(iter(loader))
    video = preprocess_front_video(batch["front_video"].to(device).float())
    labels = batch["item_label"].to(device)
    logits = classifier(front_encoder(video)["z_front_clip"])
    finite = torch.isfinite(logits).all().item()
    print(f"Front val batch: video {tuple(video.shape)}, logits {tuple(logits.shape)}, participants={batch['participant_id']}, finite={finite}")
    if (labels >= 0).any():
        acc = (logits.argmax(dim=-1)[labels >= 0] == labels[labels >= 0]).float().mean().item()
        print(f"Front val one-batch acc: {acc:.3f}")


@torch.no_grad()
def run_evaluation(val_loader, device, dual_swin, fusion, latent, decoder, evaluator):
    for m in (dual_swin, fusion, latent, decoder):
        m.eval()
    evaluator.reset()
    for batch in tqdm(val_loader, desc="Eval", dynamic_ncols=True, mininterval=30):
        j_r, j_l, r_gt, l_gt, _ = forward_ego(batch, device, dual_swin, fusion, latent, decoder)
        evaluator.update(j_r, j_l, r_gt, l_gt)
    res = evaluator.compute()
    print(f"Eval ({res['num_frames']} frames): MPJPE {res['MPJPE']:.2f} (R {res['MPJPE_R']:.2f} / L {res['MPJPE_L']:.2f})")
    return res


def load_pose_encoder(args, device):
    if not args.pose_ckpt or not os.path.exists(args.pose_ckpt):
        print("Pose checkpoint not found; ego training will skip pose-latent alignment.")
        return None
    ckpt = torch.load(args.pose_ckpt, map_location=device)
    pose_encoder = PoseEncoder().to(device)
    pose_encoder.load_state_dict(ckpt.get("pose_encoder", ckpt))
    print(f"Loaded frozen PoseEncoder from {args.pose_ckpt}")
    return pose_encoder


def build_ego_modules(device, seq_len, use_epipolar_geometry):
    if not use_epipolar_geometry:
        print("WARNING: real ASLHand2 ZED calibration not available; using geometry-free stereo cross-attention.")
    return (
        maybe_parallel(DualSwinFPN(), device),
        maybe_parallel(MultiScaleCrossViewFusion(stages=4, dim=128, pe_feats=32, heads=4, use_epipolar_geometry=use_epipolar_geometry), device),
        maybe_parallel(LatentProcessingModule(seq_len=seq_len), device),
        maybe_parallel(Decoder(mano_path), device),
        maybe_parallel(StudentProjectionHead(), device),
    )


def run_pose(args, device, writer):
    loader, ds = make_aslhand2_loader(args, split="train", max_windows=2 if args.smoke else None)
    print(f"Pose AE representation dimension: {POSE_REP_DIM}")
    print(f"Pose train participants: {ds.participants}; windows={len(ds)}")
    pose_ae = PoseAutoencoder().to(device)
    criterion = PoseAutoencoderLoss()
    optimizer = Adam(pose_ae.parameters(), lr=lr)
    best_loss, global_step = float("inf"), 0
    for epoch in range(1, args.num_epochs + 1):
        loss, global_step = train_pose_epoch(loader, device, pose_ae, criterion, optimizer, writer, global_step)
        ckpt = {"epoch": epoch, "pose_encoder": pose_ae.encoder.state_dict(), "pose_autoencoder": pose_ae.state_dict()}
        torch.save(ckpt, os.path.join(save_dir, "pose_autoencoder_last.pth"))
        if loss < best_loss:
            best_loss = loss
            torch.save(ckpt, args.pose_ckpt)
            print(f"New best pose AE loss {best_loss:.6f} -> {args.pose_ckpt}")
        if args.smoke:
            break


def run_ego(args, device, writer):
    if args.train_mode == "baseline":
        train_tars, val_tars = glob_tars(args.data_train), glob_tars(args.data_val)
        train_set = StructSyncClipDataset(train_tars, clip_len=args.clip_len, stride=stride, crop_hands=crop_hands, crop_size=img_size)
        val_set = StructSyncClipDataset(val_tars, clip_len=args.clip_len, stride=args.clip_len, crop_hands=crop_hands, crop_size=img_size)
        n_shards = lambda tars: len(tars) if isinstance(tars, list) else args.num_workers
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=min(args.num_workers, n_shards(train_tars)))
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=min(args.num_workers, n_shards(val_tars)))
        use_pose_align = False
    else:
        train_loader, train_ds = make_aslhand2_loader(args, split="train", max_windows=2 if args.smoke else None)
        val_loader, val_ds = make_aslhand2_loader(args, split="val", max_windows=1 if args.smoke else None, shuffle=False)
        print(f"ASLHand2 train participants: {train_ds.participants}; windows={len(train_ds)}")
        print(f"ASLHand2 val participants: {val_ds.participants}; windows={len(val_ds)}")
        use_pose_align = True

    modules = build_ego_modules(device, args.clip_len, str_to_bool(args.use_epipolar_geometry))
    pose_encoder = load_pose_encoder(args, device) if use_pose_align else None
    optimizer = Adam([p for m in modules for p in m.parameters() if p.requires_grad], lr=lr)
    evaluator = PoseEvaluator()
    global_step, best_mpjpe = 0, float("inf")
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n=== Ego Epoch {epoch} ===")
        mean_mpjpe, global_step = train_ego_epoch(train_loader, device, modules, pose_encoder, optimizer, writer, global_step, evaluator, use_pose_align)
        print(f"Train MPJPE: {mean_mpjpe:.2f}")
        res = run_evaluation(val_loader, device, *modules[:4], evaluator)
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "val": res,
            "swin": modules[0].state_dict(),
            "fusion": modules[1].state_dict(),
            "latent": modules[2].state_dict(),
            "decoder": modules[3].state_dict(),
            "student_projection": modules[4].state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(save_dir, "EgoSSA_last.pth"))
        if res["MPJPE"] < best_mpjpe:
            best_mpjpe = res["MPJPE"]
            torch.save(ckpt, os.path.join(save_dir, "EgoSSA_best.pth"))
        if args.smoke:
            break


def run_front(args, device, writer):
    loader, ds = make_repair_loader(args, split="train", max_samples=2 if args.smoke else None)
    val_loader, val_ds = make_repair_loader(args, split="val", max_samples=1 if args.smoke else None, shuffle=False)
    print(f"ASL Repair train participants: {ds.participants}; clips={len(ds)}")
    print(f"ASL Repair val participants: {val_ds.participants}; clips={len(val_ds)}")
    front_encoder = FrontVideoEncoder(pretrained_backbone=args.pretrained_front_backbone, freeze_backbone=args.freeze_front_backbone).to(device)
    classifier = nn.Linear(256, max(len(ds.item_to_idx), 1)).to(device)
    optimizer = Adam(list(front_encoder.parameters()) + list(classifier.parameters()), lr=lr)
    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        loss, acc, global_step = train_front_epoch(loader, device, front_encoder, classifier, optimizer, writer, global_step)
        run_front_validation_batch(val_loader, device, front_encoder, classifier)
        torch.save({"epoch": epoch, "front_encoder": front_encoder.state_dict(), "front_classifier": classifier.state_dict(), "item_to_idx": ds.item_to_idx}, os.path.join(save_dir, "front_semantic_last.pth"))
        print(f"Front epoch {epoch}: cls_loss={loss:.4f}, acc={acc:.3f}")
        if args.smoke:
            break


def run_joint_unpaired(args, device, writer):
    ego_loader, ego_ds = make_aslhand2_loader(args, split="train", max_windows=2 if args.smoke else None)
    front_loader, repair_ds = make_repair_loader(args, split="train", max_samples=2 if args.smoke else None)
    print(f"Joint ASLHand2 train participants: {ego_ds.participants}; windows={len(ego_ds)}")
    print(f"Joint ASL Repair train participants: {repair_ds.participants}; clips={len(repair_ds)}")
    modules = build_ego_modules(device, args.clip_len, str_to_bool(args.use_epipolar_geometry))
    pose_encoder = load_pose_encoder(args, device)
    front_encoder = FrontVideoEncoder(pretrained_backbone=args.pretrained_front_backbone, freeze_backbone=args.freeze_front_backbone).to(device)
    classifier = nn.Linear(256, max(len(repair_ds.item_to_idx), 1)).to(device)
    params = [p for m in modules for p in m.parameters() if p.requires_grad] + list(front_encoder.parameters()) + list(classifier.parameters())
    optimizer = Adam(params, lr=lr)
    evaluator = PoseEvaluator()
    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        global_step = train_joint_unpaired_epoch(ego_loader, front_loader, device, modules, pose_encoder, front_encoder, classifier, optimizer, writer, global_step, evaluator)
        torch.save({"epoch": epoch, "swin": modules[0].state_dict(), "fusion": modules[1].state_dict(), "latent": modules[2].state_dict(), "decoder": modules[3].state_dict(), "student_projection": modules[4].state_dict(), "front_encoder": front_encoder.state_dict(), "front_classifier": classifier.state_dict(), "item_to_idx": repair_ds.item_to_idx}, os.path.join(save_dir, "joint_unpaired_last.pth"))
        if args.smoke:
            break


def train_joint_unpaired_epoch(ego_loader, front_loader, device, modules, pose_encoder, front_encoder, classifier, optimizer, writer, global_step, evaluator):
    front_iter = iter(front_loader)
    dual_swin, fusion, latent, decoder, student_proj = modules
    loss_module = StructureLatentLoss(lambda_dyn=1.0, lambda_gate=lambda_gate)
    for m in modules:
        m.train()
    front_encoder.train()
    classifier.train()
    if pose_encoder is not None:
        pose_encoder.eval()
        pose_encoder.requires_grad_(False)

    for ego_batch in tqdm(ego_loader, desc="Joint unpaired", dynamic_ncols=True, mininterval=30):
        try:
            front_batch = next(front_iter)
        except StopIteration:
            front_iter = iter(front_loader)
            front_batch = next(front_iter)
        optimizer.zero_grad()
        lambda_kl = linear_schedule(global_step, 0.0, kl_max, kl_anneal_steps)
        unwrap(latent).tau = linear_schedule(global_step, 2.0, 0.5, tau_anneal_steps)
        j_r, j_l, r_gt, l_gt, aux = forward_ego(ego_batch, device, dual_swin, fusion, latent, decoder)
        ego_pose = ((j_r - r_gt).norm(dim=-1).mean() + (j_l - l_gt).norm(dim=-1).mean()) * 500.0
        latent_loss, _ = loss_module(aux, lambda_kl)
        align_loss = j_r.new_tensor(0.0)
        if pose_encoder is not None:
            with torch.no_grad():
                z_pose_gt = pose_encoder(stack_pose_gt(ego_batch, device))
            align_loss = F.smooth_l1_loss(student_proj(aux), z_pose_gt.detach())
        labels = front_batch["item_label"].to(device)
        valid = labels >= 0
        front_loss = j_r.new_tensor(0.0)
        if valid.any():
            video = preprocess_front_video(front_batch["front_video"].to(device).float())
            logits = classifier(front_encoder(video)["z_front_clip"])
            front_loss = F.cross_entropy(logits[valid], labels[valid]) * lambda_front_cls
        total = ego_pose + latent_loss + lambda_pose_align * align_loss + front_loss
        total.backward()
        optimizer.step()
        metrics = evaluator.evaluate(j_r, j_l, r_gt, l_gt)
        if writer:
            writer.add_scalar("Joint/total", total.item(), global_step)
            writer.add_scalar("Joint/ego_pose", ego_pose.item(), global_step)
            writer.add_scalar("Joint/pose_align", align_loss.item(), global_step)
            writer.add_scalar("Joint/front_cls", front_loss.item(), global_step)
            writer.add_scalar("Joint/mpjpe", (metrics["MPJPE_R"] + metrics["MPJPE_L"]) / 2.0, global_step)
        global_step += 1
    return global_step


def main():
    global lambda_pose_align, lambda_front_cls
    args = parse_args()
    lambda_pose_align = args.lambda_pose_align
    lambda_front_cls = args.lambda_front_cls
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    if args.train_mode in ("pose", "ego", "joint_unpaired"):
        report_calibration_state(args)
    if args.train_mode == "pose":
        run_pose(args, device, writer)
    elif args.train_mode in ("baseline", "ego"):
        run_ego(args, device, writer)
    elif args.train_mode == "front":
        run_front(args, device, writer)
    elif args.train_mode == "joint_unpaired":
        run_joint_unpaired(args, device, writer)


if __name__ == "__main__":
    main()
