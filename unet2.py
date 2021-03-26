import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """CBL: Conv -> BN -> LeakyReLU
        CL: Conv -> LeakyReLU
        CBR: Conv->BN->ReLU
    """

    def __init__(self, inChannels, outChannels, midChannels=None, kernelSize=5, layerType="CBL"):
        super(ConvLayer, self).__init__()
        if not midChannels:
            midChannels = outChannels
        if layerType == "CBL":
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=inChannels, out_channels=midChannels, kernel_size=kernelSize, padding=2),
                nn.BatchNorm2d(num_features=midChannels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        elif layerType == "CL":
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=inChannels, out_channels=midChannels, kernel_size=kernelSize, padding=2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        elif layerType == "CBR":
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=inChannels, out_channels=midChannels, kernel_size=kernelSize, padding=2),
                nn.BatchNorm2d(num_features=midChannels),
                nn.ReLU(inplace=True)
            )
        else:
            NotImplementedError(self.__class__.__init__)

    def forward(self, input_):
        return self.layer(input_)


class Upsample(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize=5, bilinear=True, layerType="CBL"):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvLayer(inChannels, outChannels, layerType=layerType)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, inChannels, outChannels, maxpool=True, layerType="CBR"):
        super(Downsample, self).__init__()
        self.isMaxpool = maxpool
        self.convLayer = ConvLayer(inChannels, outChannels, layerType=layerType)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, input_):
        return self.maxpool(self.convLayer(input_))


class UNet(nn.Module):
    def __init__(self, nChannels, nClasses):
        super(UNet, self).__init__()
        self.nChannels = nChannels
        self.nClasses = nClasses
        self.DCT = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.5)

        self.down1 = Downsample(nChannels, 64, layerType="CL")
        self.down2 = Downsample(64, 128, layerType="CBL")
        self.down3 = Downsample(128, 256, layerType="CBL")
        self.down4 = Downsample(256, 512, layerType="CBL")
        self.down5 = Downsample(512, 512, layerType="CBL")
        self.down6 = Downsample(512, 512, layerType="CBL")
        self.down7 = Downsample(512, 512, layerType="CBL")
        self.down8 = Downsample(512, 512, layerType="CBR")
        self.up1 = Upsample(1024, 512)
        self.up2 = Upsample(1024, 512)
        self.up3 = Upsample(1024, 512)
        self.up4 = Upsample(1024, 256)
        self.up5 = Upsample(512, 128)
        self.up6 = Upsample(256, 64)
        self.up7 = Upsample(128, 128)

    def forward(self, x):
        x1 = self.down1(x)  # 128x128x64
        x2 = self.down2(x1)  # 64x64x128
        x3 = self.down3(x2)  # 32x32x256
        x4 = self.down4(x3)  # 16x16x512
        x5 = self.down5(x4)  # 8x8x512
        x6 = self.down6(x5)  # 4x4x512
        x7 = self.down7(x6)  # 2x2x512
        x8 = self.down8(x7)  # 1x1x512
        x = self.up1(x8, x7)  # 2x2x1024
        x = self.up2(x, x6)  # 4x4x1024
        x = self.dropout(x)
        x = self.up3(x, x5)  # 8x8x1024
        x = self.dropout(x)
        x = self.up4(x, x4)
        x = self.dropout(x)
        x = self.up5(x, x3)  # 32x32x256
        x = self.up6(x, x2)  # 64x64x128
        x = self.dropout(x)
        x = self.up7(x, x1)  # 128x128x128
        output = self.DCT(x)  # 1x128x128
        return output
