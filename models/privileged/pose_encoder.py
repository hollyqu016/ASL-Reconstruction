import torch
import torch.nn as nn


POSE_REP_DIM = 21 * 3 + 21 * 3 + 3


def make_bimanual_pose_representation(left_joints, right_joints):
    """Encode articulation plus relative wrist displacement.

    Inputs are metric joints with shape [B,T,21,3]. The representation is:
    left local hand, right local hand, and right_wrist - left_wrist.
    """
    left_wrist = left_joints[..., :1, :]
    right_wrist = right_joints[..., :1, :]
    left_local = left_joints - left_wrist
    right_local = right_joints - right_wrist
    wrist_delta = (right_wrist - left_wrist).squeeze(-2)
    return torch.cat([left_local.flatten(-2), right_local.flatten(-2), wrist_delta], dim=-1)


class PoseEncoder(nn.Module):
    def __init__(self, input_dim=POSE_REP_DIM, hidden_dim=512, latent_dim=256, num_layers=2, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, pose_repr):
        if pose_repr.dim() != 3:
            raise ValueError(f"Expected pose representation [B,T,{self.input_dim}], got {tuple(pose_repr.shape)}")
        return self.out(self.temporal(self.input(pose_repr)))


class PoseDecoder(nn.Module):
    def __init__(self, output_dim=POSE_REP_DIM, hidden_dim=512, latent_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class PoseAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = PoseEncoder(latent_dim=latent_dim)
        self.decoder = PoseDecoder(latent_dim=latent_dim)

    def forward(self, pose_repr):
        z = self.encoder(pose_repr)
        recon = self.decoder(z)
        return {"z_pose": z, "pose_recon": recon}
