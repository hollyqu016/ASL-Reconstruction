import torch
import torch.nn as nn


class PoseEncoder(nn.Module):
    def __init__(self, num_hands=2, num_joints=21, in_dim=3, hidden_dim=512, latent_dim=256, num_layers=2, num_heads=4):
        super().__init__()
        self.num_hands = num_hands
        self.num_joints = num_joints
        self.in_dim = in_dim
        flat_dim = num_hands * num_joints * in_dim
        self.input = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
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

    def forward(self, joints):
        if joints.dim() != 5:
            raise ValueError(f"Expected joints [B,T,H,J,3], got {tuple(joints.shape)}")
        B, T = joints.shape[:2]
        x = joints.reshape(B, T, -1)
        return self.out(self.temporal(self.input(x)))


class PoseDecoder(nn.Module):
    def __init__(self, num_hands=2, num_joints=21, out_dim=3, hidden_dim=512, latent_dim=256):
        super().__init__()
        self.num_hands = num_hands
        self.num_joints = num_joints
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_hands * num_joints * out_dim),
        )

    def forward(self, z):
        joints = self.net(z)
        return joints.reshape(*z.shape[:-1], self.num_hands, self.num_joints, self.out_dim)


class PoseAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = PoseEncoder(latent_dim=latent_dim)
        self.decoder = PoseDecoder(latent_dim=latent_dim)

    def forward(self, joints):
        z = self.encoder(joints)
        recon = self.decoder(z)
        return {"z_pose": z, "pose_recon": recon}
