import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import *


class ThreeBandFrequencyGenerator(nn.Module):
    def __init__(self, low_cutoff=0.3, mid_cutoff=0.7):
        super().__init__()
        self.low_cutoff = low_cutoff
        self.mid_cutoff = mid_cutoff

    def forward(self, x):
        """
            Args:
                x: [B, C, H, W]

            Returns:
                F_low: [B, C, H, W]
                F_mid: [B, C, H, W]
                F_high: [B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device

        X = dct_2d(x)

        mask_low = build_frequency_mask(H, W, 0.0, self.low_cutoff, device)
        mask_mid = build_frequency_mask(H, W, self.low_cutoff, self.mid_cutoff, device)
        mask_high = build_frequency_mask(H, W, self.mid_cutoff, 1.0, device)

        mask_low = mask_low.unsqueeze(0).unsqueeze(0)
        mask_mid = mask_mid.unsqueeze(0).unsqueeze(0)
        mask_high = mask_high.unsqueeze(0).unsqueeze(0)

        X_low = X * mask_low
        X_mid = X * mask_mid
        X_high = X * mask_high

        F_low = idct_2d(X_low)
        F_mid = idct_2d(X_mid)
        F_high = idct_2d(X_high)

        return F_low, F_mid, F_high


class ThreeBandCP(nn.Module):
    def __init__(self, channels, reduction=16, k=16):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d((k, k))
        self.gmp = nn.AdaptiveMaxPool2d((k, k))

        self.conv_gap_low = self._make_conv(channels, reduction)
        self.conv_gmp_low = self._make_conv(channels, reduction)

        self.conv_gap_mid = self._make_conv(channels, reduction)
        self.conv_gmp_mid = self._make_conv(channels, reduction)

        self.conv_gap_high = self._make_conv(channels, reduction)
        self.conv_gmp_high = self._make_conv(channels, reduction)

        self.conv_fuse = nn.Conv2d(
            6 * (channels // reduction),
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

    def forward(self, F_low, F_mid, F_high, Ci):
        gap_low = self.conv_gap_low(self.gap(F_low)).sum(dim=(2, 3), keepdim=True)
        gmp_low = self.conv_gmp_low(self.gmp(F_low)).sum(dim=(2, 3), keepdim=True)

        gap_mid = self.conv_gap_mid(self.gap(F_mid)).sum(dim=(2, 3), keepdim=True)
        gmp_mid = self.conv_gmp_mid(self.gmp(F_mid)).sum(dim=(2, 3), keepdim=True)

        gap_high = self.conv_gap_high(self.gap(F_high)).sum(dim=(2, 3), keepdim=True)
        gmp_high = self.conv_gmp_high(self.gmp(F_high)).sum(dim=(2, 3), keepdim=True)

        c = torch.cat([gap_low, gmp_low, gap_mid, gmp_mid, gap_high, gmp_high], dim=1)

        u_cp = self.sigmoid(self.conv_fuse(c))

        return Ci * u_cp


class ThreeBandSP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_low = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.conv_mid = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.conv_high = nn.Conv2d(channels, 1, kernel_size=1, bias=False)

        self.conv_fuse = nn.Conv2d(3, 1, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, F_low, F_mid, F_high, Ci):
        u_low = self.conv_low(F_low)
        u_mid = self.conv_mid(F_mid)
        u_high = self.conv_high(F_high)

        u_concat = torch.cat([u_low, u_mid, u_high], dim=1)
        u_sp = self.sigmoid(self.conv_fuse(u_concat))

        return Ci * u_sp


class TaskSpecificThreeBandHFP(nn.Module):
    def __init__(self, channels, task_type='heatmap'):
        super().__init__()
        self.task_type = task_type

        if task_type == 'heatmap':
            self.freq_gen = ThreeBandFrequencyGenerator(
                low_cutoff=0.15,
                mid_cutoff=0.5
            )
            # self.freq_weights = [0.5, 0.4, 0.1]

        elif task_type == 'offset':
            self.freq_gen = ThreeBandFrequencyGenerator(
                low_cutoff=0.3,
                mid_cutoff=0.7
            )
            # self.freq_weights = [0.1, 0.3, 0.6]

        elif task_type == 'density':
            self.freq_gen = ThreeBandFrequencyGenerator(
                low_cutoff=0.1,
                mid_cutoff=0.3
            )
            # self.freq_weights = [0.7, 0.2, 0.1]

        self.freq_logits = nn.Parameter(torch.zeros(3))

        self.cp = ThreeBandCP(channels)
        self.sp = ThreeBandSP(channels)
        self.conv_fuse = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, Ci):
        # F_low, F_mid, F_high = self.freq_gen(Ci)
        #
        # w = torch.softmax(self.freq_logits, dim=0)
        # F = w[0] * F_low + w[1] * F_mid + w[2] * F_high
        #
        # Ci_freq = Ci + F
        #
        # C_cp = self.cp(F, F, F, Ci_freq)
        # C_sp = self.sp(F, F, F, Ci_freq)
        #
        # C_enhanced = self.conv_fuse(C_cp + C_sp)
        #
        # return Ci + 0.1 * C_enhanced

        F_low, F_mid, F_high = self.freq_gen(Ci)

        C_cp = self.cp(F_low, F_mid, F_high, Ci)
        C_sp = self.sp(F_low, F_mid, F_high, Ci)

        C_enhanced = self.conv_fuse(C_cp + C_sp)

        return 0.1 * C_enhanced + 0.9 * Ci