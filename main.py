import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.hot3d import StructSyncClipDataset
from evaluator import PoseEvaluator
from loss import StructureLatentLoss, TeacherDistillationLoss
from models.DualSwinFeatureExtractor import DualSwinFPN
from models.FrontTeacher import FrontCameraTeacher, StudentTeacherProjection
from models.LatentProcessingModule import LatentProcessingModule
from models.MANODecoder import Decoder
from models.MultiScaleCrossViewFusion import MultiScaleCrossViewFusion
from models.mano import WRIST_IDX

data_tar_train = os.environ.get("DATA_TRAIN", "data/hot3d_wds/train")
data_tar_val = os.environ.get("DATA_VAL", "data/hot3d_wds/val")
mano_path = os.environ.get("MANO_PATH", "/mnt/bigdata/data/body_models/mano")
log_dir = os.environ.get("LOG_DIR", "runs/EgoSSA")
save_dir = os.environ.get("SAVE_DIR", "checkpoints")
clip_len = 8
stride = int(os.environ.get("STRIDE", 1))
batch_size = int(os.environ.get("BATCH_SIZE", 4))
num_workers = int(os.environ.get("NUM_WORKERS", 4))
num_epochs = int(os.environ.get("NUM_EPOCHS", 3))
crop_hands = os.environ.get("CROP_HANDS", "1") == "1"
lr = 1e-4
lambda_pose = 1.0
lambda_teacher_pose = float(os.environ.get("LAMBDA_TEACHER_POSE", 0.5))
lambda_distill = float(os.environ.get("LAMBDA_DISTILL", 0.1))
kl_max = 0.01
kl_anneal_steps = 5000
tau_anneal_steps = 10000
lambda_gate = 1e-3
img_size = 224
front_img_size = int(os.environ.get("FRONT_IMG_SIZE", img_size))
use_front_teacher = os.environ.get("USE_FRONT_TEACHER", "1") == "1"
front_in_chans = int(os.environ.get("FRONT_IN_CHANS", 3))
front_max_cameras = int(os.environ.get("FRONT_MAX_CAMERAS", 8))
seed = 42

IMG_MEAN, IMG_STD = 0.449, 0.226


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def preprocess_front_images(x):
    B, T, V, C, H, W = x.shape
    x = x.reshape(B * T * V, C, H, W)
    if (H, W) != (front_img_size, front_img_size):
        x = F.interpolate(x, size=(front_img_size, front_img_size), mode="bilinear", align_corners=False)
    if front_in_chans == 1 and C == 3:
        x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    elif front_in_chans == 3 and C == 1:
        x = x.expand(-1, 3, -1, -1)
    x = (x - IMG_MEAN) / IMG_STD
    return x.view(B, T, V, front_in_chans, front_img_size, front_img_size)


def linear_schedule(step, start, end, num_steps):
    return end if step >= num_steps else start + (end - start) * step / num_steps


def build_camera_matrices(batch, device):
    intrinsics_clip = batch["intrinsics_clip"]
    extrinsics_clip = batch["extrinsics_clip"]
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
            rot, trans = extr["rotation_matrix"], extr["translation"][0]
            for r in range(3):
                for c in range(3):
                    E[t, :, r, c] = rot[r][c].float()
                E[t, :, r, 3] = trans[r].float()
            E[t, :, 3, 3] = 1.0
    K_left_inv = torch.linalg.inv(K_left)
    T_left2right = E_right @ torch.linalg.inv(E_left)
    return K_left.transpose(0, 1), K_right.transpose(0, 1), K_left_inv.transpose(0, 1), T_left2right.transpose(0, 1)


def first_existing(batch, names):
    for name in names:
        if name in batch:
            return batch[name]
    return None


def get_front_images(batch, device):
    front = first_existing(batch, ("front_img_clip", "front_imgs_clip", "front_images_clip", "front_rgb_clip"))
    if front is None:
        return None
    if isinstance(front, (list, tuple)):
        front = torch.stack(front, dim=2)
    front = front.to(device).float()
    if front.dim() != 6:
        raise ValueError(f"Front images must have shape [B,T,V,C,H,W], got {tuple(front.shape)}")
    return preprocess_front_images(front)


def get_optional_tensor(batch, names, device, dtype=torch.float32):
    value = first_existing(batch, names)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=dtype)


def triangulate_points_dlt(points_2d, confidence, intrinsics, extrinsics):
    B, T, V, J, _ = points_2d.shape
    K = intrinsics.float()
    E = extrinsics.float()
    P = K.reshape(B * T * V, 3, 3) @ E[..., :3, :].reshape(B * T * V, 3, 4)
    P = P.view(B, T, V, 3, 4)
    xy = points_2d.float()
    conf = confidence.float().clamp_min(0.0)

    rows_x = xy[..., 0, None] * P[:, :, :, None, 2, :] - P[:, :, :, None, 0, :]
    rows_y = xy[..., 1, None] * P[:, :, :, None, 2, :] - P[:, :, :, None, 1, :]
    rows = torch.stack([rows_x, rows_y], dim=-2)
    rows = rows * conf[..., None, None].clamp_min(1e-6).sqrt()
    A = rows.permute(0, 1, 3, 2, 4, 5).reshape(B * T * J, V * 2, 4)

    _, _, vh = torch.linalg.svd(A)
    homog = vh[:, -1]
    denom = torch.where(homog[:, 3:].abs() < 1e-8, homog[:, 3:].sign().clamp_min(0.0) * 2e-8 - 1e-8, homog[:, 3:])
    xyz = homog[:, :3] / denom
    return xyz.view(B, T, J, 3)


def get_front_pose_pseudo(batch, device):
    pseudo_r = get_optional_tensor(batch, ("front_pseudo_right_landmarks_clip", "teacher_right_landmarks_clip"), device)
    pseudo_l = get_optional_tensor(batch, ("front_pseudo_left_landmarks_clip", "teacher_left_landmarks_clip"), device)
    if pseudo_r is not None and pseudo_l is not None:
        return pseudo_r / 1000.0 if pseudo_r.abs().mean() > 10 else pseudo_r, pseudo_l / 1000.0 if pseudo_l.abs().mean() > 10 else pseudo_l

    pts_r = get_optional_tensor(batch, ("front_right_keypoints_2d_clip", "front_right_2d_clip"), device)
    pts_l = get_optional_tensor(batch, ("front_left_keypoints_2d_clip", "front_left_2d_clip"), device)
    K = get_optional_tensor(batch, ("front_intrinsics_clip", "front_K_clip"), device)
    E = get_optional_tensor(batch, ("front_extrinsics_clip", "front_T_world_from_cam_clip", "front_projection_extrinsics_clip"), device)
    if pts_r is None or pts_l is None or K is None or E is None:
        return None, None

    conf_r = get_optional_tensor(batch, ("front_right_keypoint_confidence_clip", "front_right_confidence_clip"), device)
    conf_l = get_optional_tensor(batch, ("front_left_keypoint_confidence_clip", "front_left_confidence_clip"), device)
    if conf_r is None:
        conf_r = torch.ones_like(pts_r[..., 0])
    if conf_l is None:
        conf_l = torch.ones_like(pts_l[..., 0])
    return triangulate_points_dlt(pts_r, conf_r, K, E), triangulate_points_dlt(pts_l, conf_l, K, E)


def forward_model(batch, device, dual_swin, fusion, latent, decoder, front_teacher=None, student_proj=None):
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

    teacher_out = None
    if front_teacher is not None and student_proj is not None:
        front_images = get_front_images(batch, device)
        if front_images is not None:
            camera_ids = get_optional_tensor(batch, ("front_camera_ids", "front_camera_ids_clip"), device, dtype=torch.long)
            view_conf = get_optional_tensor(batch, ("front_view_confidence_clip", "front_confidence_clip"), device)
            teacher_out = front_teacher(front_images, camera_ids=camera_ids, view_confidence=view_conf)
            teacher_out["student_repr"] = student_proj(aux)
            pseudo_r, pseudo_l = get_front_pose_pseudo(batch, device)
            if pseudo_r is not None and pseudo_l is not None:
                teacher_out["pseudo_r"] = to_relative(pseudo_r.float())
                teacher_out["pseudo_l"] = to_relative(pseudo_l.float())

    return to_relative(j_r), to_relative(j_l), r_gt, l_gt, aux, teacher_out


def unwrap(module):
    return module.module if hasattr(module, "module") else module


def train_one_epoch(train_loader, device, dual_swin, fusion, latent, decoder, front_teacher, student_proj, loss_module, distill_loss_module, optimizer, evaluator, writer, global_step):
    for m in (dual_swin, fusion, latent, decoder, front_teacher, student_proj):
        if m is None:
            continue
        m.train()
    mpjpe_r_all, mpjpe_l_all = [], []
    pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True, mininterval=30)
    for batch in pbar:
        lambda_kl = linear_schedule(global_step, 0.0, kl_max, kl_anneal_steps)
        unwrap(latent).tau = linear_schedule(global_step, 2.0, 0.5, tau_anneal_steps)
        optimizer.zero_grad()
        j_r, j_l, r_gt, l_gt, aux, teacher_out = forward_model(batch, device, dual_swin, fusion, latent, decoder, front_teacher, student_proj)
        pose_loss = ((j_r - r_gt).norm(dim=-1).mean() + (j_l - l_gt).norm(dim=-1).mean()) * 1000.0 / 2
        latent_loss, loss_dict = loss_module(aux, lambda_kl)
        teacher_pose_loss = j_r.new_tensor(0.0)
        distill_loss = j_r.new_tensor(0.0)
        if teacher_out is not None:
            distill_loss = distill_loss_module(teacher_out["student_repr"], teacher_out["teacher_repr"])
            if "pseudo_r" in teacher_out and "pseudo_l" in teacher_out:
                teacher_pose_loss = (
                    (j_r - teacher_out["pseudo_r"].detach()).norm(dim=-1).mean()
                    + (j_l - teacher_out["pseudo_l"].detach()).norm(dim=-1).mean()
                ) * 1000.0 / 2
        total_loss = lambda_pose * pose_loss + latent_loss + lambda_teacher_pose * teacher_pose_loss + lambda_distill * distill_loss
        total_loss.backward()
        optimizer.step()
        metrics = evaluator.evaluate(j_r, j_l, r_gt, l_gt)
        if writer:
            for k, v in metrics.items():
                writer.add_scalar(f"Metrics/{k}", v, global_step)
            writer.add_scalar("Loss/Total", total_loss.item(), global_step)
            writer.add_scalar("Loss/Joint", pose_loss.item(), global_step)
            writer.add_scalar("Loss/TeacherPose", teacher_pose_loss.item(), global_step)
            writer.add_scalar("Loss/Distill", distill_loss.item(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"Loss/{k}", v, global_step)
            writer.add_scalar("Sched/lambda_kl", lambda_kl, global_step)
        pbar.set_postfix({"L_joint": pose_loss.item(), "KL": loss_dict["L_kl_prior"], "MPJPE_R": metrics["MPJPE_R"], "MPJPE_L": metrics["MPJPE_L"]})
        mpjpe_r_all.append(metrics["MPJPE_R"])
        mpjpe_l_all.append(metrics["MPJPE_L"])
        global_step += 1
    return (np.mean(mpjpe_r_all) + np.mean(mpjpe_l_all)) / 2, global_step


@torch.no_grad()
def run_evaluation(val_loader, device, dual_swin, fusion, latent, decoder, evaluator):
    for m in (dual_swin, fusion, latent, decoder):
        m.eval()
    evaluator.reset()
    for batch in tqdm(val_loader, desc="Eval", dynamic_ncols=True, mininterval=30):
        j_r, j_l, r_gt, l_gt, _, _ = forward_model(batch, device, dual_swin, fusion, latent, decoder)
        evaluator.update(j_r, j_l, r_gt, l_gt)
    res = evaluator.compute()
    print(f"Eval ({res['num_frames']} frames): MPJPE {res['MPJPE']:.2f} (R {res['MPJPE_R']:.2f} / L {res['MPJPE_L']:.2f}) | "
          f"PA-MPJPE {res['PA-MPJPE']:.2f} | PCK@5 {res['PCK@5']:.2f} | PCK@10 {res['PCK@10']:.2f} | AUC@30 {res['AUC@30']:.2f}")
    return res


def main():
    set_seed(seed)
    device = torch.device("cuda")
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(save_dir, exist_ok=True)

    train_tars, val_tars = glob_tars(data_tar_train), glob_tars(data_tar_val)
    train_set = StructSyncClipDataset(train_tars, clip_len=clip_len, stride=stride, crop_hands=crop_hands, crop_size=img_size)
    val_set = StructSyncClipDataset(val_tars, clip_len=clip_len, stride=clip_len, crop_hands=crop_hands, crop_size=img_size)
    n_shards = lambda tars: len(tars) if isinstance(tars, list) else num_workers
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=min(num_workers, n_shards(train_tars)))
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=min(num_workers, n_shards(val_tars)))

    dual_swin = torch.nn.DataParallel(DualSwinFPN().to(device))
    fusion = torch.nn.DataParallel(MultiScaleCrossViewFusion(stages=4, dim=128, pe_feats=32, heads=4).to(device))
    latent = torch.nn.DataParallel(LatentProcessingModule(seq_len=clip_len).to(device))
    decoder = torch.nn.DataParallel(Decoder(mano_path).to(device))
    front_teacher = None
    student_proj = None
    if use_front_teacher:
        front_teacher = torch.nn.DataParallel(FrontCameraTeacher(in_chans=front_in_chans, max_cameras=front_max_cameras).to(device))
        student_proj = torch.nn.DataParallel(StudentTeacherProjection().to(device))

    train_modules = [m for m in (dual_swin, fusion, latent, decoder, front_teacher, student_proj) if m is not None]
    params = [p for m in train_modules for p in m.parameters()]
    optimizer = Adam(params, lr=lr)
    loss_module = StructureLatentLoss(lambda_dyn=1.0, lambda_gate=lambda_gate)
    distill_loss_module = TeacherDistillationLoss()
    evaluator = PoseEvaluator()

    global_step = 0
    best_mpjpe = float("inf")
    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch} ===")
        mean_mpjpe, global_step = train_one_epoch(
            train_loader, device, dual_swin, fusion, latent, decoder, front_teacher, student_proj,
            loss_module, distill_loss_module, optimizer, evaluator, writer, global_step
        )
        print(f"Train MPJPE: {mean_mpjpe:.2f}")
        res = run_evaluation(val_loader, device, dual_swin, fusion, latent, decoder, evaluator)
        for k, v in res.items():
            writer.add_scalar(f"Val/{k}", v, epoch)
        ckpt = {"epoch": epoch, "global_step": global_step, "val": res, "swin": dual_swin.state_dict(), "fusion": fusion.state_dict(),
                "latent": latent.state_dict(), "decoder": decoder.state_dict(), "optimizer": optimizer.state_dict()}
        if front_teacher is not None and student_proj is not None:
            ckpt["front_teacher"] = front_teacher.state_dict()
            ckpt["student_proj"] = student_proj.state_dict()
        torch.save(ckpt, os.path.join(save_dir, "EgoSSA_last.pth"))
        if res["MPJPE"] < best_mpjpe:
            best_mpjpe = res["MPJPE"]
            torch.save(ckpt, os.path.join(save_dir, "EgoSSA_best.pth"))
            print(f"New best val MPJPE {best_mpjpe:.2f} (epoch {epoch})")
    print(f"Best val MPJPE: {best_mpjpe:.2f}")


if __name__ == "__main__":
    main()
