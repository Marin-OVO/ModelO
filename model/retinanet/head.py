import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinaPointHead(nn.Module):
    """
        RetinaNet-style point prediction head
        Input: FPN features (P3–P7), each (B, C, H, W)
        Output: point heatmap per level (B, 1, H, W)
    """
    def __init__(
        self,
        in_channels=256,
        num_classes=2,
        num_convs=4,
        prior_prob=0.01
    ):
        super().__init__()

        tower = []
        for _ in range(num_convs):
            tower.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True)
            )
            tower.append(nn.ReLU(inplace=True))
        self.point_tower = nn.Sequential(*tower)

        self.point_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, padding=1
        )

        # RetinaNet-style bias init
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.point_logits.bias, bias_value)

        for m in self.point_tower.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.point_logits.weight, std=0.01)

    def forward(self, features):
        """
            features: list[Tensor], FPN features
            return: list[Tensor], point heatmaps
        """
        # fpn map -> heatmap
        outputs = []
        for feat in features:
            x = self.point_tower(feat)
            logits = self.point_logits(x)
            outputs.append(logits)

        return outputs
