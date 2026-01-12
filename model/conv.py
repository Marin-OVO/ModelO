import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLu(nn.Module):
    "conv + bn + relu"
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int=3, dilation: int=1, with_bn: bool=True):
        super().__init__()
        self.with_bn = with_bn
        bias = not with_bn

        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch) if self.with_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLu):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int=3, dilation: int=1, with_bn: bool=True, with_down: bool=True):
        super().__init__(in_ch, out_ch, kernel_size, dilation, with_bn)
        self.with_down = with_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_down:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=False)

        return super().forward(x)


class UpConvBNReLu(ConvBNReLu):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int=3, dilation: int=1, with_bn: bool=True, with_up: bool=True):
        super().__init__(in_ch, out_ch, kernel_size, dilation, with_bn)
        self.with_up = with_up

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.with_up:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=False)

        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


if __name__ == '__main__':
    pass