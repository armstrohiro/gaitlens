"""Inference architecture for the project's YOLO26n 23-point checkpoint.

Derived from gait_analysis_100subjects/src/foot23_models.py. The training loss
is intentionally omitted; module names and state-dict layout are preserved.
"""
import copy

from torch import nn
from ultralytics import YOLO


class BodyFootConv(nn.Module):
    def __init__(self, body, channels_per_joint):
        super().__init__()
        self.body = body
        self.foot = nn.Conv2d(body.in_channels, 6 * channels_per_joint, 1,
                              bias=body.bias is not None).to(body.weight.device)

    def forward(self, x):
        import torch
        return torch.cat((self.body(x), self.foot(x)), dim=1)


def extend_model(weights, device="cpu"):
    model = copy.deepcopy(YOLO(str(weights)).model).to(device)
    head = model.model[-1]
    if head.__class__.__name__ != "Pose26" or list(head.kpt_shape) != [17, 3]:
        raise ValueError("追加学習モデルにはYOLO26 Poseの17関節基盤重みが必要です。")
    for branch in ("cv4_kpts", "one2one_cv4_kpts"):
        setattr(head, branch, nn.ModuleList(BodyFootConv(conv, 3) for conv in getattr(head, branch)))
    for branch in ("cv4_sigma", "one2one_cv4_sigma"):
        setattr(head, branch, nn.ModuleList(BodyFootConv(conv, 2) for conv in getattr(head, branch)))
    head.kpt_shape = [23, 3]
    head.nk = 69
    head.nk_sigma = 46
    model.yaml = copy.deepcopy(model.yaml)
    model.yaml["kpt_shape"] = [23, 3]
    return model

