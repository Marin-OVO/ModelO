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

        self.up_3_2 = (Up(256, 128))
        self.up_5_4 = (Up(512, 512))
        self.up_x3_to_x1 = (Up(2, 2))
        self.up_x5_to_x1 = (Up(2, 2))

        self.relu = nn.ReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1)
        self.conv1x1_ = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1)

        self.x3_x2_block_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.x5_x4_block_conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)

        self.heatmap_head = HeatmapHead(in_channels=64, out_channels=num_class)
        self.heatmap_head_x3 = HeatmapHead(in_channels=128, out_channels=num_class)
        self.heatmap_head_x5 = HeatmapHead(in_channels=512, out_channels=num_class)
        self.density_predictor = DensityPredictor(in_channels=64, out_channels=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, Ci, mask=None):
        x1 = self.inc(Ci)   # 64
        x2 = self.down1(x1) # 128  /2
        x3 = self.down2(x2) # 256  /4  @@
        x4 = self.down3(x3) # 512  /8
        x5 = self.down4(x4) # 512  /16 @@

        x3_to_x2 = self.up_3_2(x3, x2) # 128 /2
        x5_to_x4 = self.up_5_4(x5, x4) # 512 /8

        x3_x2_mid = torch.cat([x3_to_x2, x2], dim=1) # 256  /2
        x3_x2_block = self.x3_x2_block_conv(x3_x2_mid)      # 128  /2
        x5_x4_mid = torch.cat([x5_to_x4, x4], dim=1) # 1024 /8
        x5_x4_block = self.x5_x4_block_conv(x5_x4_mid)      # 512  /8

        x3_sub_pool = subtract_avg_pool(x3_x2_block)
        x3_sub_pool_relu = self.relu(x3_sub_pool)
        x5_sub_pool = subtract_avg_pool(x5_x4_block)
        x5_sub_pool_relu = self.relu(x5_sub_pool)

        x3_out = x3_x2_block + x3_x2_block * self.sigmoid(x3_sub_pool)
        x5_out = x5_x4_block + x5_x4_block * self.sigmoid(x5_sub_pool)

        Fi = self.up1(x5, x4)
        Fi = self.up2(Fi, x3)
        Fi = self.up3(Fi, x2)
        Fi = self.up4(Fi, x1)

        density_out = self.density_predictor(Fi)  # [B,1,H,W]
        density_feat = self.conv1x1_(density_out)  # 64
        Fi_sub = Fi * density_feat

        heatmap_out = self.heatmap_head(Fi + Fi_sub)
        x3_map_out = self.heatmap_head_x3(x3_out)
        x5_map_out = self.heatmap_head_x5(x5_out)

        x3_map_out = self.up_x3_to_x1(x3_map_out, x1)
        x5_map_out = self.up_x5_to_x1(x5_map_out, x1)

        return {
            "heatmap_out": heatmap_out,
            "x3_map_out": x3_map_out,
            "x5_map_out": x5_map_out,
            "x3": x3_sub_pool_relu,
            "x5": x5_sub_pool_relu
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