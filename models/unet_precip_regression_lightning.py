from models.unet_parts import Down, DoubleConv, Up, OutConv
from models.unet_parts_depthwise_separable import DoubleConvDS, UpDS,  DownDS, DoubleShuffledConvDS, UpShuffledDS, DownShuffledDS
from models.layers import CBAM
from models.regression_lightning import Precip_regression_base
#from models.cloud_regression_lightning import Cloud_base
from models.layers import eca_layer as ECA
#from ddf import DDFPack
import torch.nn.functional as F
from torch import nn
import torch
from models.shuffle_attention import sa_layer as SA


class SSA_UNet(Precip_regression_base):
    def __init__(self, hparams):
        super(SSA_UNet, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = True
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        kernels_per_layer_up = self.hparams.kernels_per_layer_up
        pw_groups = 16
        print(kernels_per_layer)

        self.inc = DoubleShuffledConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer  ,  )
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownShuffledDS(64, 128, kernels_per_layer=kernels_per_layer  , pw_groups=16)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownShuffledDS(128, 256, kernels_per_layer=kernels_per_layer , pw_groups=16)
        self.cbam3 = SA(256)
        self.down3 = DownShuffledDS(256, 512, kernels_per_layer=kernels_per_layer , pw_groups=32)
        self.cbam4 = SA(512)
        factor = 2 if self.bilinear else 1
        self.down4 = DownShuffledDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer  , pw_groups=32)
        self.cbam5 = SA(1024 // factor)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer_up, )
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer_up, )
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer_up,)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer_up, )

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
    
class SSA_UNetRed(Precip_regression_base):
    def __init__(self, hparams):
        super(SSA_UNetRed, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = True
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        kernels_per_layer_up = self.hparams.kernels_per_layer_up 
        pw_groups = 16
        print(kernels_per_layer)

        self.inc = DoubleShuffledConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer  ,  )
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownShuffledDS(64, 128, kernels_per_layer=kernels_per_layer  , pw_groups=16)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownShuffledDS(128, 256, kernels_per_layer=kernels_per_layer , pw_groups=16)
        self.cbam3 = SA(256)
        self.down3 = DownShuffledDS(256, 512, kernels_per_layer=kernels_per_layer , pw_groups=32)
        self.cbam4 = SA(512)
        factor = 2 if self.bilinear else 1
        self.down4 = DownShuffledDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer  , pw_groups=32)
        self.cbam5 = SA(1024 // factor)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer_up, )
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer_up, )
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer_up,)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer_up, )

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class UNetDSShuffle_Attention32F(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention32F, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = SA(64, groups=32)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = SA(256, groups=32)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = SA(512, groups=32)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = SA(1024 // factor, groups=32)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
    
class UNetDSShuffle_Attention64F(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention64F, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = SA(64, groups=32)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = SA(128, groups=64)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = SA(256, groups=64)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = SA(512, groups=64)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = SA(1024 // factor, groups=64)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
    
class UNetDSShuffle_Attention4G(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention4G, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = SA(256, groups=64)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = SA(512, groups=128)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = SA(1024 // factor, groups=256)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
    
class UNetDSShuffle_Attention8G(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention8G, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = SA(64, groups=8)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = SA(128, groups=16)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = SA(256, groups=32)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = SA(512, groups=64)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = SA(1024 // factor, groups=128)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
    
class UNetDSShuffle_Attention32FV2(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention32FV2, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = SA(64, groups=32)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = SA(256, groups=32)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = SA(512, groups=32)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = SA(1024 // factor, groups=32)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
    
class UNetDSShuffle_Attention32FRed(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention32FRed, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = 3
        pw_groups = 16
        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = SA(64, groups=32)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer, pw_groups=pw_groups)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer, pw_groups=pw_groups)
        self.cbam3 = SA(256, groups=32)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer, pw_groups=pw_groups)
        self.cbam4 = SA(512, groups=32)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer - 1, pw_groups=pw_groups)
        self.cbam5 = SA(1024 // factor, groups=32)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer- 1)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer- 1)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer- 1)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer- 1)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
    

class UNetDSShuffle_Attention2Red(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention2Red, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        pw_groups = 16

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer +1,)
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam2 = SA(128)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam3 = SA(256)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam4 = SA(512)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer, pw_groups=pw_groups)
        self.cbam5 = SA(1024 // factor)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
    
class UNetDSShuffle_Attention3RedV2(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention3RedV2, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        pw_groups = 16

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer +1,)
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam3 = SA(256)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam4 = SA(512)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam5 = SA(1024 // factor)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class UNetDSShuffle_Attention16FRed(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention16FRed, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = 3
        pw_groups = 16

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer, pw_groups=pw_groups)
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer, pw_groups=pw_groups)
        self.cbam2 = SA(128, groups=16)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer, pw_groups=pw_groups)
        self.cbam3 = SA(256, groups=16)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer, pw_groups=pw_groups)
        self.cbam4 = SA(512, groups=16)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer - 1, pw_groups=pw_groups)
        self.cbam5 = SA(1024 // factor, groups=16)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer - 1)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer - 1)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer - 1)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer - 1)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class UNetDSShuffle_Attention16F(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention16F, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = SA(128, groups=16)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = SA(256, groups=16)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = SA(512, groups=16)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = SA(1024 // factor, groups=16)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


    def __init__(self, hparams):
        super(UNetDSShuffle_AttentionLast12, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = SA(128, groups=16)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = SA(256, groups=32)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = SA(512, groups=64)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = SA(1024 // factor, groups=64)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class UNetDSShuffle_Attention3RedV212O(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention3RedV212O, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        pw_groups = 16

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer +1,)
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam3 = SA(256)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam4 = SA(512)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam5 = SA(1024 // factor)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
    

class UNetDSShuffle_Attention3RedV26O(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention3RedV26O, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        pw_groups = 16

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer +1,)
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam3 = SA(256)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam4 = SA(512)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam5 = SA(1024 // factor)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class UNetDSShuffle_Attention4RedV26O(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention4RedV26O, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        pw_groups = 16

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer ,)
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer , pw_groups=pw_groups)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer , pw_groups=pw_groups)
        self.cbam3 = SA(256)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer , pw_groups=pw_groups*2)
        self.cbam4 = SA(512)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer , pw_groups=pw_groups*2)
        self.cbam5 = SA(1024 // factor)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

'''class UNetDSShuffle_Attention4RedV2Cloud(Cloud_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention4RedV2Cloud, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        pw_groups = 16

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer ,)
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer , pw_groups=pw_groups)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer , pw_groups=pw_groups)
        self.cbam3 = SA(256)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer , pw_groups=pw_groups*2)
        self.cbam4 = SA(512)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer , pw_groups=pw_groups*2)
        self.cbam5 = SA(1024 // factor)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class UNetDSShuffle_Attention3RedV2Cloud(Cloud_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention3RedV2Cloud, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        pw_groups = 16

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer +1,)
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam3 = SA(256)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam4 = SA(512)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer +1, pw_groups=pw_groups)
        self.cbam5 = SA(1024 // factor)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class UNetDS_AttentionCloud(Cloud_base):
    def __init__(self, hparams):
        super(UNetDS_AttentionCloud, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, ddf=True, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, ddf=True,  kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, ddf=True ,kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, ddf=True , kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

        
    
    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits'''

class UNetDSShuffle_Attention5(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention5, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        pw_groups = 16

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer +1 ,)
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer +1 , pw_groups=pw_groups)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer +1 , pw_groups=pw_groups)
        self.cbam3 = SA(256)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer +1 , pw_groups=pw_groups*2)
        self.cbam4 = SA(512)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer +1 , pw_groups=pw_groups*2)
        self.cbam5 = SA(1024 // factor)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class UNetDSShuffle_Attention2V2(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDSShuffle_Attention2V2, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels  , 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = SA(64, groups=16)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = SA(128, groups=32)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = SA(256)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = SA(512)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = SA(1024 // factor)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits