import torch.nn as nn

class MiddleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        middle_out = self.conv(x)

        return middle_out


class HeatmapHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        heatmap_out = self.conv(x)

        return heatmap_out


class DensityPredictor(nn.Module):
    def __init__(self, in_channels, out_channels: int=1, mid_channels: int=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, mid_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4, mid_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Softplus()
        )

    def forward(self, x):
        density_out = self.conv(x)

        return density_out
