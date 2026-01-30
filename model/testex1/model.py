from .conv import *
from .head import DensityPredictor, HeatmapHead, MiddleConv


class UNet1(nn.Module):
    def __init__(self, in_channels, num_class=2, bilinear=False):
        super(UNet1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = num_class
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (DownScaling(64, 128))
        self.down2 = (DownScaling(128, 256))
        self.down3 = (DownScaling(256, 512))

        factor = 2 if bilinear else 1
        self.down4 = (DownScaling(512, 1024 // factor))
        self.up1 = (UpScaling(1024, 512 // factor, bilinear))
        self.up2 = (UpScaling(512, 256 // factor, bilinear))
        self.up3 = (UpScaling(256, 128 // factor, bilinear))
        self.up4 = (UpScaling(128, 64, bilinear))

        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1)

        # self.middle_conv = MiddleConv(in_channels=64, out_channels=32)
        self.heatmap_head = HeatmapHead(in_channels=64, out_channels=num_class)
        self.density_predictor = DensityPredictor(in_channels=64, out_channels=1)

        self.conv1x1_ = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1)

    def forward(self, Ci, mask=None):
        x1 = self.inc(Ci)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        Fi = self.up1(x5, x4)
        Fi = self.up2(Fi, x3)
        Fi = self.up3(Fi, x2)
        Fi = self.up4(Fi, x1)

        density_out = self.density_predictor(Fi)  # [B,1,H,W]

        if mask is not None:
            density_out = density_out * mask

        density_feat = self.conv1x1_(density_out)  # 64

        Fi_sub = Fi * density_feat
        heatmap_out = self.heatmap_head(Fi + Fi_sub)

        return {
            "heatmap_out": heatmap_out,
            "density_out": density_out
        }

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)