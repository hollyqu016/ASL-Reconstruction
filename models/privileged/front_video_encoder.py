import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    def __init__(self, in_chans=3, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class FrontVideoEncoder(nn.Module):
    def __init__(self, in_chans=3, frame_dim=256, latent_dim=256, num_layers=2, num_heads=4):
        super().__init__()
        self.frame_encoder = FrameEncoder(in_chans=in_chans, feature_dim=frame_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, frame_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=frame_dim,
            nhead=num_heads,
            dim_feedforward=frame_dim * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, latent_dim))

    def forward(self, video):
        if video.dim() != 5:
            raise ValueError(f"Expected front video [B,T,3,H,W], got {tuple(video.shape)}")
        B, T, C, H, W = video.shape
        frame_feats = self.frame_encoder(video.reshape(B * T, C, H, W)).reshape(B, T, -1)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = self.temporal(torch.cat([cls, frame_feats], dim=1))
        z_clip = self.proj(tokens[:, 0])
        return {"z_front_clip": z_clip, "front_frame_features": tokens[:, 1:]}
