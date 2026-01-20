import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import TaskSpecificThreeBandHFP


class HeatmapHead(nn.Module):
    """
        heatmap head
    """
    def __init__(self, in_channels, out_channels, use_freq: bool=False):
        super().__init__()
        self.use_freq = use_freq
        if use_freq:
            self.freq_enhance = TaskSpecificThreeBandHFP(in_channels, task_type='heatmap')

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.use_freq:
            x = self.freq_enhance(x)

        heatmap_out = self.conv(x)

        return heatmap_out


class OffsetHead(nn.Module):
    """
       offset head

       Return:
           (B, 2, H, W) offset-x -> x, offset-y -> y
    """
    def __init__(self, in_channels, hidden_channels: int=128, out_channels: int=2, use_freq: bool=False):
        super().__init__()
        self.use_freq = use_freq
        if use_freq:
            self.freq_enhance = TaskSpecificThreeBandHFP(in_channels, task_type='offset')

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.Tanh()
        )

    def forward(self, x):
        if self.use_freq:
            x = self.freq_enhance(x)

        offset_out = self.conv(x)

        return offset_out


class DensityHead(nn.Module):
    """
        density head

        Return:
            (B, 1, H, W) dense map -> counting
    """
    def __init__(self, in_channels, hidden_channels: int=128, out_channels: int=1, use_freq: bool=False):
        super().__init__()
        self.use_freq = use_freq
        if use_freq:
            self.freq_enhance = TaskSpecificThreeBandHFP(in_channels, task_type='density')

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.Softplus()
        )

    def forward(self, x):
        if self.use_freq:
            x = self.freq_enhance(x)

        density_out =self.conv(x)

        return density_out
