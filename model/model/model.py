from .conv import *


class YNet(nn.Module):
    """
        Args:
            num_ch: num channels
            num_class: num classes
    """
    def __init__(self, num_ch, num_class=2, bilinear=False):
        super(YNet, self).__init__()
        self.n_channels = num_ch
        self.n_classes = num_class
        self.bilinear = bilinear

        self.inc = (DoubleConv(num_ch, 64))
        self.down1 = (DownScaling(64, 128))
        self.down2 = (DownScaling(128, 256))
        self.down3 = (DownScaling(256, 512))

        factor = 2 if bilinear else 1
        self.down4 = (DownScaling(512, 1024 // factor))
        self.up1 = (UpScaling(1024, 512 // factor, bilinear))
        self.up2 = (UpScaling(512, 256 // factor, bilinear))
        self.up3 = (UpScaling(256, 128 // factor, bilinear))
        self.up4 = (UpScaling(128, 64, bilinear))
        self.outc = nn.Sequential(
            nn.Conv2d(64, num_class, kernel_size=1),
            nn.Sigmoid()
        )

        self.up4_y = UpScaling(128, 64, bilinear)

        self.youtc = nn.Sequential(
            DoubleConv(64, 128),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.sigma_head = nn.Sequential(
            OutConv(64, 1),
            nn.Softplus()
        )

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4_y(y, x1)

        y = self.youtc(y)
        mu = self.mu_head(y)
        sigma = self.sigma_head(y) + 1e-3

        return logits, mu, sigma

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
        self.outc = torch.utils.checkpoint(self.outc)

        self.youtc = torch.utils.checkpoint(self.youtc)
        self.mu_head = torch.utils.checkpoint(self.mu_head)
        self.sigma_head = torch.utils.checkpoint(self.sigma_head)