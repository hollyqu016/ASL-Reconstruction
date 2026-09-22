import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SharedSwinFPN(nn.Module):
    def __init__(self, in_chans=3, out_dims=(96, 192, 384, 768), dim=128, pretrained=True):
        super().__init__()
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
        )
        self.fpn_convs = nn.ModuleList([nn.Conv2d(d, dim, kernel_size=1) for d in out_dims])

    def forward(self, images):
        feats = self.swin(images)
        return [self.fpn_convs[i](f.permute(0, 3, 1, 2)) for i, f in enumerate(feats)]


class ConfidenceAwareViewAttention(nn.Module):
    def __init__(self, dim=128, repr_dim=256, num_stages=4, max_cameras=8, dropout=0.1):
        super().__init__()
        self.camera_embed = nn.Embedding(max_cameras, dim)
        self.score_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, 1),
            )
            for _ in range(num_stages)
        ])
        self.repr_proj = nn.Sequential(
            nn.Linear(dim * num_stages, repr_dim),
            nn.LayerNorm(repr_dim),
            nn.GELU(),
            nn.Linear(repr_dim, repr_dim),
        )

    def forward(self, view_feats, camera_ids=None, view_confidence=None, front_valid=None):
        fused_feats, pooled_feats, attn_weights = [], [], []
        for stage_idx, feat in enumerate(view_feats):
            B, T, V, C, H, W = feat.shape
            pooled = F.adaptive_avg_pool2d(feat.reshape(B * T * V, C, H, W), (1, 1)).view(B, T, V, C)

            if camera_ids is None:
                ids = torch.arange(V, device=feat.device).view(1, 1, V).expand(B, T, V)
            else:
                ids = camera_ids.to(feat.device).long()
                if ids.dim() == 1:
                    ids = ids.view(1, 1, V).expand(B, T, V)
                elif ids.dim() == 2:
                    ids = ids.view(B, 1, V).expand(B, T, V)
            ids = ids.clamp(max=self.camera_embed.num_embeddings - 1)
            token = pooled + self.camera_embed(ids)

            score = self.score_heads[stage_idx](token).squeeze(-1)
            if view_confidence is not None:
                conf = view_confidence.to(feat.device).float().clamp_min(1e-6)
                score = score + conf.log()
            if front_valid is not None:
                valid = front_valid.to(feat.device).bool()
                score = score.masked_fill(~valid, -1e9)
            weights = torch.softmax(score, dim=-1)
            fused = (feat * weights[..., None, None, None]).sum(dim=2)

            fused_feats.append(fused)
            pooled_feats.append((pooled * weights[..., None]).sum(dim=2))
            attn_weights.append(weights)

        teacher_repr = self.repr_proj(torch.cat(pooled_feats, dim=-1))
        return {
            "fused_feats": fused_feats,
            "teacher_repr": teacher_repr,
            "view_weights": attn_weights,
        }


class FrontCameraTeacher(nn.Module):
    def __init__(
        self,
        in_chans=3,
        dim=128,
        repr_dim=256,
        max_cameras=8,
        pretrained=True,
    ):
        super().__init__()
        self.encoder = SharedSwinFPN(in_chans=in_chans, dim=dim, pretrained=pretrained)
        self.fusion = ConfidenceAwareViewAttention(
            dim=dim,
            repr_dim=repr_dim,
            num_stages=len(self.encoder.fpn_convs),
            max_cameras=max_cameras,
        )

    def forward(self, front_images, camera_ids=None, view_confidence=None, front_valid=None):
        B, T, V, C, H, W = front_images.shape
        images = front_images.reshape(B * T * V, C, H, W)
        encoded = self.encoder(images)
        view_feats = [f.view(B, T, V, *f.shape[1:]) for f in encoded]
        return self.fusion(
            view_feats,
            camera_ids=camera_ids,
            view_confidence=view_confidence,
            front_valid=front_valid,
        )


class TeacherPoseHead(nn.Module):
    def __init__(self, repr_dim=256, keypoint_dim=3, num_joints=21, dropout=0.1):
        super().__init__()
        self.keypoint_dim = keypoint_dim
        self.num_joints = num_joints
        self.head = nn.Sequential(
            nn.Linear(repr_dim, repr_dim),
            nn.LayerNorm(repr_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(repr_dim, 2 * num_joints * keypoint_dim),
        )

    def forward(self, teacher_repr):
        B, T, _ = teacher_repr.shape
        pred = self.head(teacher_repr).view(B, T, 2, self.num_joints, self.keypoint_dim)
        return {
            "teacher_left_joints": pred[:, :, 0],
            "teacher_right_joints": pred[:, :, 1],
        }


class StudentTeacherProjection(nn.Module):
    def __init__(self, latent_dim=256, repr_dim=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim * 2, repr_dim),
            nn.LayerNorm(repr_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(repr_dim, repr_dim),
        )

    def forward(self, aux):
        student = torch.cat([aux["mu_L"], aux["mu_R"]], dim=-1)
        return self.proj(student)
