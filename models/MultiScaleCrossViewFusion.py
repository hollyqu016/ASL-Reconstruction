import torch
import torch.nn as nn
import torch.nn.functional as F


class EpiPositionalEncoding(nn.Module):
    def __init__(self, num_feats=32):
        super().__init__()
        self.num_feats = num_feats

    def forward(self, grid, K, K_inv, T_lr):
        B, T, H, W, _ = grid.shape
        N = H * W
        uv_norm = grid.reshape(B, T, N, 2)
        uv_pix = (uv_norm + 1) * torch.tensor([W, H], device=grid.device) / 2
        ones = torch.ones(B, T, N, 1, device=grid.device)
        uv1 = torch.cat([uv_pix, ones], dim=-1).reshape(B * T, N, 3).transpose(1, 2)
        cam = K_inv.reshape(B * T, 3, 3) @ uv1
        R = T_lr[:, :, :3, :3].reshape(B * T, 3, 3)
        t = T_lr[:, :, :3, 3:].reshape(B * T, 3, 1)
        cam_t = R @ cam + t
        epi = F.normalize(cam_t - cam, dim=1).transpose(1, 2)
        pe_list = []
        for i in range(self.num_feats):
            div = 10000 ** (2 * (i // 2) / self.num_feats)
            pe_list.append(torch.sin(epi / div))
            pe_list.append(torch.cos(epi / div))
        pe = torch.cat(pe_list, dim=-1)
        return pe.reshape(B, T, H, W, -1)


class CrossViewDeformableAttention(nn.Module):
    def __init__(self, dim, pe_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.pe_proj = nn.Linear(pe_dim, dim)

    def forward(self, lf, rf, pe=None):
        B, T, C, H, W = lf.shape
        N = H * W
        q = lf.reshape(B * T, C, N).transpose(1, 2)
        k = rf.reshape(B * T, C, N).transpose(1, 2)
        if pe is not None:
            pe_mapped = self.pe_proj(pe.reshape(B * T, N, -1))
            q_in, k_in = q + pe_mapped, k + pe_mapped
        else:
            q_in, k_in = q, k
        attn_out, _ = self.attn(q_in, k_in, k)
        out = self.norm(attn_out + q)
        return out.transpose(1, 2).reshape(B, T, C, H, W)


class MultiScaleCrossViewFusion(nn.Module):
    def __init__(self, stages=4, dim=128, pe_feats=32, heads=4, use_epipolar_geometry=True):
        super().__init__()
        self.use_epipolar_geometry = use_epipolar_geometry
        self.pe_modules = nn.ModuleList([EpiPositionalEncoding(pe_feats) for _ in range(stages)])
        pe_dim = 3 * 2 * pe_feats
        self.attn_modules = nn.ModuleList([CrossViewDeformableAttention(dim, pe_dim, heads) for _ in range(stages)])

    def forward(self, left_feats, right_feats, K, K_inv, T_lr):
        fused = []
        for i, (lf, rf) in enumerate(zip(left_feats, right_feats)):
            B, T, C, H, W = lf.shape
            pe = None
            if self.use_epipolar_geometry:
                if K is None or K_inv is None or T_lr is None:
                    raise ValueError("Epipolar fusion requires real camera calibration. Use --use_epipolar_geometry false when calibration is unavailable.")
                grid = F.affine_grid(
                    torch.eye(2, 3, device=lf.device).unsqueeze(0).repeat(B * T, 1, 1),
                    size=(B * T, C, H, W), align_corners=False
                ).view(B, T, H, W, 2)
                pe = self.pe_modules[i](grid, K, K_inv, T_lr)
            fused.append(self.attn_modules[i](lf, rf, pe))
        return fused
