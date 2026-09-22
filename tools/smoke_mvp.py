import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.FrontTeacher import FrontCameraTeacher, StudentTeacherProjection, TeacherPoseHead


def assert_finite(name, x):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} contains NaN/Inf")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, V, C, H, W = 1, 2, 1, 3, 224, 224
    D, J, K = 256, 21, 3

    front_rgb = torch.randn(B, T, V, C, H, W, device=device)
    front_valid = torch.ones(B, T, V, dtype=torch.bool, device=device)
    gt_left = torch.randn(B, T, J, K, device=device)
    gt_right = torch.randn(B, T, J, K, device=device)

    teacher = FrontCameraTeacher(pretrained=False).to(device)
    pose_head = TeacherPoseHead(keypoint_dim=K).to(device)
    teacher_out = teacher(front_rgb, front_valid=front_valid)
    pred = pose_head(teacher_out["teacher_repr"])
    teacher_loss = (
        F.smooth_l1_loss(pred["teacher_left_joints"], gt_left)
        + F.smooth_l1_loss(pred["teacher_right_joints"], gt_right)
    )
    teacher_loss.backward()
    assert_finite("teacher_loss", teacher_loss)

    proj = StudentTeacherProjection().to(device)
    aux = {
        "mu_L": torch.randn(B, T, D, device=device, requires_grad=True),
        "mu_R": torch.randn(B, T, D, device=device, requires_grad=True),
    }
    student_repr = proj(aux)
    distill_loss = F.smooth_l1_loss(student_repr, teacher_out["teacher_repr"].detach())
    distill_loss.backward()
    assert_finite("distill_loss", distill_loss)

    print("MVP smoke OK")
    print(f"front_rgb: {tuple(front_rgb.shape)}")
    print(f"teacher_repr: {tuple(teacher_out['teacher_repr'].shape)}")
    print(f"teacher_left_joints: {tuple(pred['teacher_left_joints'].shape)}")
    print(f"student_projected_repr: {tuple(student_repr.shape)}")
    print(f"teacher_loss: {teacher_loss.item():.6f}")
    print(f"distill_loss: {distill_loss.item():.6f}")


if __name__ == "__main__":
    main()
