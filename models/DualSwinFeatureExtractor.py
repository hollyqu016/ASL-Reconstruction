import timm
import torch.nn as nn


class DualSwinFPN(nn.Module):
    def __init__(self, in_chans=1, out_dims=(96, 192, 384, 768), dim=128, pretrained=True):
        super().__init__()
        self.swin_left = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, in_chans=in_chans, features_only=True)
        self.swin_right = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, in_chans=in_chans, features_only=True)
        self.fpn_convs = nn.ModuleList([nn.Conv2d(d, dim, kernel_size=1) for d in out_dims])

    def _encode(self, swin, img, B, T):
        feats = swin(img)
        feats = [self.fpn_convs[i](f.permute(0, 3, 1, 2)) for i, f in enumerate(feats)]
        return [f.view(B, T, *f.shape[1:]) for f in feats]

    def forward(self, left_img, right_img):
        B, T, C, H, W = left_img.shape
        left = self._encode(self.swin_left, left_img.view(B * T, C, H, W), B, T)
        right = self._encode(self.swin_right, right_img.view(B * T, C, H, W), B, T)
        return left, right
