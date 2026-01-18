import torch
import torch.nn as nn
import torch.nn.functional as F


class OffsetHead(nn.Module):
    """
       offset head

       Return:
           (B, 2, H, W) offset-x -> x, offset-y -> y
    """
    def __init__(self, in_channels, hidden_channels: int=128, out_channels: int=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1)
        )

    def forward(self, x):
        offset_out = self.conv(x)

        return offset_out


class DensityHead(nn.Module):
    """
        density head

        Return:
            (B, 1, H, W) dense map -> counting
    """
    def __init__(self, in_channels, hidden_channels: int=128, out_channels: int=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.Softplus()
        )

    def forward(self, x):
        density_out =self.conv(x)

        return density_out
