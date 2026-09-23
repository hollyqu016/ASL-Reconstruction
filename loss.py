import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureLatentLoss(nn.Module):
    def __init__(self, lambda_dyn=1.0, lambda_gate=1e-3, kl_eps=1e-8):
        super().__init__()
        self.lambda_dyn = lambda_dyn
        self.lambda_gate = lambda_gate
        self.kl_eps = kl_eps

    def forward(self, aux, lambda_kl):
        L_kl_prior = self.kl_standard_normal(aux["mu_L"], aux["logvar_L"]) + self.kl_standard_normal(aux["mu_R"], aux["logvar_R"])
        L_kl_dyn = self.temporal_kl(aux["mu_L"], aux["logvar_L"]) + self.temporal_kl(aux["mu_R"], aux["logvar_R"])
        L_gate = torch.stack([g.abs().mean() for g in aux["gates"]]).mean()
        total = lambda_kl * (L_kl_prior + self.lambda_dyn * L_kl_dyn) + self.lambda_gate * L_gate
        return total, {"L_kl_prior": L_kl_prior.item(), "L_kl_dyn": L_kl_dyn.item(), "L_gate": L_gate.item()}

    @staticmethod
    def kl_standard_normal(mu, logvar):
        return (0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)).sum(-1).mean()

    def temporal_kl(self, mu, logvar):
        mu_t, mu_prev = mu[:, 1:], mu[:, :-1]
        logvar_t, logvar_prev = logvar[:, 1:], logvar[:, :-1]
        kl = 0.5 * (
            (logvar_prev - logvar_t)
            + (logvar_t.exp() + (mu_t - mu_prev).pow(2)) / (logvar_prev.exp() + self.kl_eps)
            - 1
        )
        return kl.sum(-1).mean()


class TeacherDistillationLoss(nn.Module):
    def __init__(self, mode="smooth_l1"):
        super().__init__()
        self.mode = mode

    def forward(self, student_repr, teacher_repr):
        teacher_repr = teacher_repr.detach()
        if self.mode == "cosine":
            return 1.0 - F.cosine_similarity(student_repr, teacher_repr, dim=-1).mean()
        if self.mode == "mse":
            return F.mse_loss(student_repr, teacher_repr)
        return F.smooth_l1_loss(student_repr, teacher_repr)


class PoseAutoencoderLoss(nn.Module):
    def __init__(self, lambda_velocity=0.05):
        super().__init__()
        self.lambda_velocity = lambda_velocity

    def forward(self, pred, target):
        recon = F.smooth_l1_loss(pred, target)
        if pred.shape[1] > 1:
            pred_vel = pred[:, 1:] - pred[:, :-1]
            target_vel = target[:, 1:] - target[:, :-1]
            velocity = F.smooth_l1_loss(pred_vel, target_vel)
        else:
            velocity = pred.new_tensor(0.0)
        total = recon + self.lambda_velocity * velocity
        return total, {"pose_recon": recon.item(), "pose_velocity": velocity.item()}
