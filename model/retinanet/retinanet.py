import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import resnet50_fpn_backbone
from .head import RetinaPointHead


class RetinaPointNet(nn.Module):
    def __init__(self, in_channels, num_classes, output_scales=['P3', 'P4', 'P5']):
        super().__init__()

        self.output_scales = output_scales

        self.backbone = resnet50_fpn_backbone(
            pretrain_path="model/retinanet/resnet50-0676ba61.pth",
            trainable_layers=3,
            in_channels=in_channels
        )

        self.head = RetinaPointHead(
            in_channels=256,
            num_classes=num_classes
        )

    def forward(self, x):
        B, _, H, W = x.shape  # image

        x = self.backbone(x)  # OrderedDict
        x = list(x.values())  # [P3, P4, P5, P6, P7]
        x = self.head(x)      # list of (B, C, Hi, Wi)

        out = {}
        upsampled_logits = []

        fpn = ['P3', 'P4', 'P5', 'P6', 'P7']
        for i, (logits, name) in enumerate(zip(x, fpn)):
            upsampled = F.interpolate(
                logits,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

            if name in self.output_scales: # ['P3', 'P4', 'P5']
                out[name] = upsampled

            if i < 3:
                upsampled_logits.append(upsampled)

        # final layer
        if len(upsampled_logits) > 0:
            out['final'] = torch.mean(torch.stack(upsampled_logits), dim=0)
        else:
            out['final'] = out[self.output_scales[0]]

        return out
