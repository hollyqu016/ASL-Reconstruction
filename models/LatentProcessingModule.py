import torch
import torch.nn as nn


def sample_latent(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std


def gumbel_sigmoid(logits, tau=1.0, eps=1e-10):
    U = torch.rand_like(logits)
    g = torch.log(U + eps) - torch.log(1 - U + eps)
    return torch.sigmoid((logits + g) / tau)


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x


class LatentProcessingModule(nn.Module):
    def __init__(self, dim_in=128, dim_latent=256, seq_len=8, num_heads=4, dropout=0.1, num_stages=4, num_anchors=4, logvar_clamp=(-8.0, 4.0)):
        super().__init__()
        self.num_anchors = num_anchors
        self.logvar_clamp = logvar_clamp
        self.tau = 2.0

        self.input_proj = nn.Sequential(
            nn.Linear(dim_in, dim_latent),
            nn.LayerNorm(dim_latent),
            GaussianNoise(std=0.1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.shared_anchor_tokens = nn.Parameter(torch.randn(num_anchors, dim_latent))
        self.time_embed = nn.Parameter(torch.zeros(1, seq_len, dim_latent))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_latent, nhead=num_heads, batch_first=True, dropout=dropout)
        self.st_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.gumbel_gate = nn.Linear(dim_latent, 1)

        self.left_mu = nn.Linear(dim_latent * num_stages, dim_latent)
        self.left_logvar = nn.Linear(dim_latent * num_stages, dim_latent)
        self.right_mu = nn.Linear(dim_latent * num_stages, dim_latent)
        self.right_logvar = nn.Linear(dim_latent * num_stages, dim_latent)

    def forward(self, fused_feats):
        stage_latents, gates = [], []
        for feats in fused_feats:
            B, T, _ = feats.shape
            x = self.input_proj(feats) + self.time_embed[:, :T]
            anchors = self.shared_anchor_tokens.unsqueeze(0).expand(B, -1, -1)
            latent_seq = self.st_transformer(torch.cat([anchors, x], dim=1))[:, self.num_anchors:]

            gate_logits = self.gumbel_gate(latent_seq)
            gate = gumbel_sigmoid(gate_logits, tau=self.tau) if self.training else torch.sigmoid(gate_logits)
            context = (latent_seq * gate).sum(dim=1, keepdim=True) / (gate.sum(dim=1, keepdim=True) + 1e-6)
            stage_latents.append(gate * latent_seq + (1 - gate) * context)
            gates.append(gate)

        z_cat = torch.cat(stage_latents, dim=-1)
        mu_L, mu_R = self.left_mu(z_cat), self.right_mu(z_cat)
        logvar_L = self.left_logvar(z_cat).clamp(*self.logvar_clamp)
        logvar_R = self.right_logvar(z_cat).clamp(*self.logvar_clamp)

        if self.training:
            z_R, z_L = sample_latent(mu_R, logvar_R), sample_latent(mu_L, logvar_L)
        else:
            z_R, z_L = mu_R, mu_L

        aux = {"mu_L": mu_L, "logvar_L": logvar_L, "mu_R": mu_R, "logvar_R": logvar_R, "gates": gates}
        return z_R, z_L, aux
