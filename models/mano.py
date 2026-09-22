import os

import smplx
import torch

JOINT_MAPPING = [16, 17, 18, 19, 20, 0, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]
FINGERTIP_VERTS = [744, 320, 443, 554, 671]
NUM_MANO_JOINTS_WITH_TIPS = 21
WRIST_IDX = 5
NUM_POSE_PCA = 15
NUM_BETAS = 10


def create_mano_layers(mano_dir):
    left = smplx.create(os.path.join(mano_dir, "MANO_LEFT.pkl"), "mano", use_pca=True, is_rhand=False, num_pca_comps=NUM_POSE_PCA)
    right = smplx.create(os.path.join(mano_dir, "MANO_RIGHT.pkl"), "mano", use_pca=True, is_rhand=True, num_pca_comps=NUM_POSE_PCA)
    if torch.sum(torch.abs(left.shapedirs[:, 0, :] - right.shapedirs[:, 0, :])) < 1:
        left.shapedirs[:, 0, :] *= -1
    return left, right


def mano_joints(layer, betas, hand_pose, global_orient, transl):
    out = layer(betas=betas, global_orient=global_orient, hand_pose=hand_pose, transl=transl, return_verts=True)
    joints = out.joints
    if joints.shape[1] != NUM_MANO_JOINTS_WITH_TIPS:
        joints = torch.cat([joints, out.vertices[:, FINGERTIP_VERTS]], dim=1)
    return joints[:, JOINT_MAPPING]
