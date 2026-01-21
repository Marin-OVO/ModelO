import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import *


def build_normalized_frequency_map(H, W, device):
    y = torch.arange(H, device=device).float()
    x = torch.arange(W, device=device).float()

    yy, xx = torch.meshgrid(y, x, indexing='ij')

    freq_map = torch.sqrt(xx ** 2 + yy ** 2)

    max_freq = torch.sqrt(torch.tensor((H - 1) ** 2 + (W - 1) ** 2, device=device))
    freq_map = freq_map / (max_freq + 1e-6)

    return freq_map.clamp(0, 1)


class LearnableFrequencyFilter(nn.Module):
    def __init__(self, channels, freq_dim=32):
        super().__init__()
        self.freq_dim = freq_dim

        self.freq_selector = nn.Sequential(
            nn.Linear(1, freq_dim),
            nn.ReLU(inplace=True),
            nn.Linear(freq_dim, freq_dim),
            nn.ReLU(inplace=True),
            nn.Linear(freq_dim, channels),
            nn.Sigmoid()
        )

        self.num_basis = 8
        self.basis_centers = nn.Parameter(torch.linspace(0, 1, self.num_basis))
        self.basis_widths = nn.Parameter(torch.ones(self.num_basis) * 0.2)
        self.basis_weights = nn.Parameter(torch.randn(channels, self.num_basis) * 0.1)

    def get_frequency_weights(self, H, W, device):
        freq_map = build_normalized_frequency_map(H, W, device)  # [H, W]

        freq_expanded = freq_map.unsqueeze(-1)  # [H, W, 1]
        centers = self.basis_centers.view(1, 1, -1)  # [1, 1, num_basis]
        widths = self.basis_widths.view(1, 1, -1)  # [1, 1, num_basis]

        basis_response = torch.exp(
            -(freq_expanded - centers) ** 2 / (2 * widths ** 2 + 1e-6)
        )  # [H, W, num_basis]

        basis_response = basis_response.permute(2, 0, 1)  # [num_basis, H, W]

        freq_weights = torch.sigmoid(
            torch.matmul(
                self.basis_weights,  # [C, num_basis]
                basis_response.reshape(self.num_basis, -1)  # [num_basis, H*W]
            ).reshape(-1, H, W)  # [C, H, W]
        )

        return freq_weights

    def forward(self, Ci):
        B, C, H, W = Ci.shape
        device = Ci.device

        X = dct_2d(Ci)  # [B, C, H, W]

        freq_weights = self.get_frequency_weights(H, W, device)  # [C, H, W]
        freq_weights = freq_weights.unsqueeze(0)  # [1, C, H, W]

        Fi = X * freq_weights
        Fi = idct_2d(Fi)

        return Fi


class CP(nn.Module):
    def __init__(self, channels, reduction=16, k=16):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d((k, k))
        self.gmp = nn.AdaptiveMaxPool2d((k, k))

        self.conv_gap = self._make_conv(channels, reduction)
        self.conv_gmp = self._make_conv(channels, reduction)

        self.conv_fuse = nn.Conv2d(
            2 * (channels // reduction),
            channels,
            kernel_size=1,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def _make_conv(self, channels, reduction):
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels // reduction,
                kernel_size=1,
                groups=channels // reduction,
                bias=False
            )
        )

    def forward(self, Fi, Ci):
        gap = self.conv_gap(self.gap(Fi)).sum(dim=(2, 3), keepdim=True)
        gmp = self.conv_gmp(self.gmp(Fi)).sum(dim=(2, 3), keepdim=True)

        c = torch.cat([gap, gmp], dim=1)
        u_cp = self.sigmoid(self.conv_fuse(c))

        return Ci * u_cp


class SP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.conv_fuse = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, Fi, Ci):
        u = self.conv(Fi)
        u_sp = self.sigmoid(u)

        return Ci * u_sp


class LearnableFrequencyHead(nn.Module):
    def __init__(self, channels, task_type='heatmap'):
        super().__init__()
        self.task_type = task_type

        self.freq_filter = LearnableFrequencyFilter(channels, freq_dim=32)
        self.cp = CP(channels)
        self.sp = SP(channels)
        self.conv_fuse = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, Ci):
        Fi = self.freq_filter(Ci)

        # Fi += Ci

        C_cp = self.cp(Fi, Ci)
        C_sp = self.sp(Fi, Ci)

        C_enhanced = self.conv_fuse(C_cp + C_sp)

        alpha = torch.sigmoid(self.alpha)
        Ci_out = alpha * C_enhanced + (1 - alpha) * Ci

        return Ci_out