import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18FrameEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True, feature_dim=256):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.pretrained = pretrained
        self.expected_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.expected_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(in_features, feature_dim)
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def forward(self, x):
        mean = self.expected_mean.to(device=x.device, dtype=x.dtype)
        std = self.expected_std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        return self.proj(self.backbone(x))


class FrontVideoEncoder(nn.Module):
    def __init__(
        self,
        frame_dim=256,
        latent_dim=256,
        num_layers=2,
        num_heads=4,
        pretrained_backbone=True,
        freeze_backbone=True,
    ):
        super().__init__()
        self.frame_encoder = ResNet18FrameEncoder(
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            feature_dim=frame_dim,
        )
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
