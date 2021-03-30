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
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample=nn.ConvTranspose2d(in_channels=inChannels, out_channels=inChannels // 2, kernel_size=5,stride=2)
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
    def __init__(self, nChannels, nClasses, dropout=0.5, activation="tanh"):
        super(UNet, self).__init__()
        self.nChannels = nChannels
        self.nClasses = nClasses

        if activation=="tanh":
            self.DCT = nn.Sequential(
                nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=1),
                nn.Tanh()
            )
        else:
            self.DCT = nn.Sequential(
                nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=1),
                nn.Sigmoid()
            )


        self.dropout = nn.Dropout(dropout)
        self.inc = ConvLayer(nChannels, 16, layerType="CL")
        self.down1 = Downsample(16, 32, layerType="CBL")
        self.down2 = Downsample(32, 64, layerType="CBL")
        self.down3 = Downsample(64, 128, layerType="CBL")
        self.down4 = Downsample(128, 256, layerType="CBL")
        self.down5 = Downsample(256, 512, layerType="CBR")
        # self.middle = Downsample(512, 512, layerType="CBR")
        self.up1 = Upsample(512, 256)
        self.up2 = Upsample(256, 128)
        self.up3 = Upsample(128, 64)
        self.up4 = Upsample(64, 32)
        self.up5 = Upsample(32, 16)
        self.up6 = Upsample(16, nClasses, activation)

    def forward(self, x):
        # 1x256x256
        x1 = self.inc(x) # 16x256x256
        x2 = self.down1(x1) # 32x128x128
        x3 = self.down2(x2) # 64x64x64
        x4 = self.down3(x3) # 128x32x32
        x5 = self.down4(x4) # 256x16x16
        x6 = self.down5(x5) # 512x8x8
        x = self.up1(x6, x5) # 256x16x16
        x=self.dropout(x)
        x = self.up2(x, x4) # 128x32x32
        x = self.dropout(x)
        x = self.up3(x, x3) # 64x64x64
        x=self.dropout(x)
        x = self.up4(x, x2) # 32x128x128
        x = self.dropout(x)
        x = self.up5(x, x1) # 16x256x256
        output = self.DCT(x)
        return output