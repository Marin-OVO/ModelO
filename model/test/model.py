"""
    unet
"""
from .conv import *
from .head import OffsetHead, DensityHead


class UNet_(nn.Module):
    """
        U-Shape, down then up
        
        Args:
            in_channels: num channels
            num_class: num classes
        Return:
            Dict {
            'heatmap_out': heatmap_out, (B, 2, H, W)
            'offset_out': offset_out,   (B, 2, H, W)
            'density_out': density_out  (B, 1, H, W)
            }
    """
    def __init__(self, in_channels, num_class=2, bilinear=False):
        super(UNet_, self).__init__()
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

        self.heatmap_head = (OutConv(64, num_class))
        self.offset_head = OffsetHead(in_channels=64, hidden_channels=128, out_channels=2)
        self.density_head = DensityHead(in_channels=64, hidden_channels=128, out_channels=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        heatmap_out = self.heatmap_head(x)
        offset_out = self.offset_head(x)
        density_out = self.density_head(x)

        return {
            'heatmap_out': heatmap_out,
            'offset_out': offset_out,
            'density_out': density_out
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