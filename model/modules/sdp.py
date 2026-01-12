import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SDP(nn.Module):
    """
        Ci       -> Q
        Pi+1     -> V, K
        Q + K    -> attn
        attn + V -> F_sdp
    """
    def __init__(self, channels, c5_size):
        super().__init__()

        self.channels = channels
        self.H5, self.W5 = c5_size

        self.conv_q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, Ci, Pi1):
        B, C, Hi, Wi = Ci.shape

        Pu = F.interpolate(Pi1, size=(Hi, Wi), mode="bilinear", align_corners=False)

        Q = self.conv_q(Ci)
        K = self.conv_k(Pu)
        V = self.conv_v(Pu)

        assert Hi % self.H5 == 0 and Wi % self.W5 == 0, \
            "Feature size must be divisible by C5 size"

        n_h = Hi // self.H5
        n_w = Wi // self.W5
        n = n_h * n_w

        def blockify(x):
            x = x.view(B, C, n_h, self.H5, n_w, self.W5)
            x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
            x = x.view(B, n, self.H5 * self.W5, C)

            return x

        Qb = blockify(Q)
        Kb = blockify(K)
        Vb = blockify(V)

        # Q, K
        aj = torch.matmul(Qb, Kb.transpose(-2, -1))
        aj = aj / math.sqrt(C)
        aj = F.softmax(aj, dim=-1)

        # attn, V
        Fb = torch.matmul(aj, Vb)

        Fb = Fb.view(B, n_h, n_w, self.H5, self.W5, C)
        Fb = Fb.permute(0, 5, 1, 3, 2, 4).contiguous()
        F_out = Fb.view(B, C, Hi, Wi)

        return Ci + F_out
