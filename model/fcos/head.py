import torch
import torch.nn as nn
import torch.nn.functional as F


class PointHead(nn.Module):
    """
        Point Head
    """
    def __init__(self, in_channels=256, num_classes=2, num_convs=4, prior_prob=0.01):
        super().__init__()

        cls_tower = []
        neighbor_tower = []

        for _ in range(num_convs):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False))
            cls_tower.append(nn.BatchNorm2d(in_channels))
            cls_tower.append(nn.ReLU(inplace=True))

            neighbor_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False))
            neighbor_tower.append(nn.BatchNorm2d(in_channels))
            neighbor_tower.append(nn.ReLU(inplace=True))

        self.cls_tower = nn.Sequential(*cls_tower)
        self.neighbor_tower = nn.Sequential(*neighbor_tower)

        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.point_logits = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self.radius_head = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # bias init
        bias = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_logits.bias, bias)

    def forward(self, x):

        # fpn map -> heatmap
        num_map = 0
        cls_outputs = []
        neighbor_outputs = []

        for features in x: # x -> list: [P2, P3, P4, P5, P6]
            cls_feat = self.cls_tower(features)
            neighbor_logits = self.neighbor_tower(features)
            neighbor_prob = torch.sigmoid(neighbor_logits)

            cls_outputs.append(self.cls_logits(cls_feat))
            neighbor_outputs.append(neighbor_prob)

            num_map += 1

        return cls_outputs, neighbor_outputs # logits, prob

