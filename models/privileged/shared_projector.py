import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentProjectionHead(nn.Module):
    def __init__(self, latent_dim=256, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, out_dim),
        )

    def forward(self, aux):
        return self.net(torch.cat([aux["mu_L"], aux["mu_R"]], dim=-1))


class SharedGestureProjector(nn.Module):
    def __init__(self, in_dim=256, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, normalize=True):
        y = self.net(x)
        return F.normalize(y, dim=-1) if normalize else y
