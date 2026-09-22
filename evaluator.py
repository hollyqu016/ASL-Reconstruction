import numpy as np
import torch


def procrustes_align(pred, gt):
    mu_p, mu_g = pred.mean(dim=1, keepdim=True), gt.mean(dim=1, keepdim=True)
    X, Y = (pred - mu_p).transpose(1, 2), (gt - mu_g).transpose(1, 2)
    var_x = (X ** 2).sum(dim=(1, 2))
    K = X @ Y.transpose(1, 2)
    U, _, Vh = torch.linalg.svd(K)
    V = Vh.transpose(1, 2)
    Z = torch.eye(3, device=pred.device, dtype=pred.dtype).repeat(pred.shape[0], 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))
    R = V @ Z @ U.transpose(1, 2)
    scale = torch.diagonal(R @ K, dim1=1, dim2=2).sum(-1) / (var_x + 1e-12)
    aligned = scale[:, None, None] * (R @ X) + mu_g.transpose(1, 2)
    return aligned.transpose(1, 2)


class PoseEvaluator:
    def __init__(self, pck_thresholds=(5.0, 10.0), auc_max=30.0, auc_step=1.0):
        self.pck_thresholds = pck_thresholds
        self.auc_thresholds = np.arange(0, auc_max + auc_step, auc_step)
        self.reset()

    def reset(self):
        self.dists = {"R": [], "L": []}
        self.pa_dists = {"R": [], "L": []}

    @staticmethod
    def _flat(x):
        return x.reshape(-1, x.shape[-2], 3).float()

    @torch.no_grad()
    def update(self, pred_r, pred_l, gt_r, gt_l):
        for side, pred, gt in (("R", pred_r, gt_r), ("L", pred_l, gt_l)):
            pred, gt = self._flat(pred), self._flat(gt)
            assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"
            self.dists[side].append(((pred - gt).norm(dim=-1) * 1000.0).cpu())
            self.pa_dists[side].append(((procrustes_align(pred, gt) - gt).norm(dim=-1) * 1000.0).cpu())

    def _metrics(self, d, pa):
        out = {"MPJPE": d.mean().item(), "PA-MPJPE": pa.mean().item()}
        for t in self.pck_thresholds:
            out[f"PCK@{t:g}"] = (d < t).float().mean().item() * 100.0
        out["AUC@30"] = np.mean([(d < t).float().mean().item() for t in self.auc_thresholds]) * 100.0
        return out

    def compute(self):
        res, per_side = {}, {}
        for side in ("R", "L"):
            d, pa = torch.cat(self.dists[side]), torch.cat(self.pa_dists[side])
            per_side[side] = (d, pa)
            for k, v in self._metrics(d, pa).items():
                res[f"{k}_{side}"] = v
        res.update(self._metrics(torch.cat([per_side["R"][0], per_side["L"][0]]), torch.cat([per_side["R"][1], per_side["L"][1]])))
        res["num_frames"] = per_side["R"][0].shape[0]
        return res

    @torch.no_grad()
    def evaluate(self, pred_r, pred_l, gt_r, gt_l):
        pred_r, pred_l, gt_r, gt_l = map(self._flat, (pred_r, pred_l, gt_r, gt_l))
        return {
            "MPJPE_R": ((pred_r - gt_r).norm(dim=-1).mean() * 1000.0).item(),
            "MPJPE_L": ((pred_l - gt_l).norm(dim=-1).mean() * 1000.0).item(),
        }
