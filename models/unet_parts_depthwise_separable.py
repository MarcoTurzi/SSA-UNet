""" Parts of the U-Net model """
# Base model taken from: https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import DepthwiseSeparableConv, ShuffledDepthwiseSeparableConv



class DoubleShuffledConvDS(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1, norm_layer=nn.BatchNorm2d, activation=nn.ReLU, pw_groups=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ShuffledDepthwiseSeparableConv(
                in_channels,
                mid_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
                groups=pw_groups
            ),
            norm_layer(mid_channels),
            activation(),
            DepthwiseSeparableConv(
                mid_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            norm_layer(out_channels),
            activation(),
        )

    def forward(self, x):
        return self.double_conv(x)

class DownShuffledDS(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1, norm_layer=nn.BatchNorm2d, activation=nn.ReLU, pw_groups=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleShuffledConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer, norm_layer=norm_layer, activation=activation, pw_groups=pw_groups),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpShuffledDS(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, ddf=False, kernels_per_layer=1, partial=False, norm_layer=nn.BatchNorm2d, activation=nn.ReLU, pw_groups=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleShuffledConvDS(
                in_channels,
                out_channels,
                in_channels // 2,
                kernels_per_layer=kernels_per_layer,
                norm_layer=norm_layer,
                activation=activation,
                pw_groups=pw_groups
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleShuffledConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer, norm_layer=norm_layer, activation=activation, pw_groups=pw_groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
    # input is CHW
    
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)

class DoubleConvDS(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1, norm_layer=nn.BatchNorm2d, activation=nn.ReLU, pw_groups=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels,
                mid_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
                groups=pw_groups
            ),
            norm_layer(mid_channels),
            activation(),
            DepthwiseSeparableConv(
                mid_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            norm_layer(out_channels),
            activation(),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownDS(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1, norm_layer=nn.BatchNorm2d, activation=nn.ReLU, pw_groups=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer, norm_layer=norm_layer, activation=activation, pw_groups=pw_groups),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, ddf=False, kernels_per_layer=1, partial=False, norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                in_channels // 2,
                kernels_per_layer=kernels_per_layer,
                norm_layer=norm_layer,
                activation=activation
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer, norm_layer=norm_layer, activation=activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
    # input is CHW
    
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)



