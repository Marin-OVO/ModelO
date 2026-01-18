import os
import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
from .fpn import BackboneWithFPN, LastLevelMaxPool


class Bottleneck(nn.Module):
    """
        Pipeline: conv1x1, bn, relu -> conv3x3, bn, relu -> conv1x1, bn, res, relu
        Channel: in_channels -> hidden_channels -> hidden_channels * expansion
        Size: size -> size/s -> size/s
    """
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # in_ch -> out_ch
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = norm_layer(out_channel)


        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = norm_layer(out_channel)

        # in_ch -> out_ch * scale
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = norm_layer(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        # input -> conv1x1 + bn + relu (B, out_ch ,H, W)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # input -> conv3x3 + bn + relu (B, out_ch ,H/s, W/s)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # input -> conv1x1 + bn + relu (B, out_ch * scale ,H/s, W/s)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out # f(x) + x


class ResNet(nn.Module):
    """
        Pipeline: conv7x7, bn, relu, pool -> layer1 -> layer2 -> layer3 -> layer4
        Channel: in_channels -> in_channels -> 2 * in_channels -> 4 * in_channels -> 8 * in_channels
        Size: size/2, size/4 -> size/4 -> size/8 -> size/16 -> size/32
    """
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True,
                 norm_layer=None, in_channels=3):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False) # /2
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.in_channel != channel * block.expansion:
            # conv + bn
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, norm_layer=norm_layer))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top: # full connection
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def overwrite_eps(model, eps):
    """
        This method overwrites the default eps values of all the
        FrozenBatchNorm2d layers of the model with the provided value.
        This is necessary to address the BC-breaking change introduced
        by the bug-fix at pytorch/vision#2933. The overwrite is applied
        only when the pretrained weights are loaded to maintain compatibility
        with previous versions.

        Args:
            model (nn.Module): The model on which we perform the overwrite.
            eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


def resnet50_fpn_backbone(pretrain_path="",
                          norm_layer=FrozenBatchNorm2d,  # FrozenBatchNorm2d -> BatchNorm2d if batch_size>>
                          trainable_layers=3,
                          returned_layers=None,
                          extra_blocks=None,
                          in_channels=3
                          ):
    """
        ResNet-50

        conv1
        layer1: 3xBottleneck(3x3)
        layer2: 4xBottleneck(4x3)
        layer3: 6xBottleneck(6x3)
        layer4: 3xBottleneck(3x3)
        fc
    """
    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3],
                             include_top=False,
                             norm_layer=norm_layer,
                             in_channels=in_channels,
                             )

    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)

    if pretrain_path != "":
        assert os.path.exists(pretrain_path), "{} is not exist.".format(pretrain_path)
        resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False) # missing key

    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    if trainable_layers == 5: # layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1']
        layers_to_train.append("bn1")

    for name, parameter in resnet_backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5

    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = resnet_backbone.in_channel // 8  # 256
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256

    # backbone: resnet50 (top: fpn, bottom: backbone) -> dict
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
