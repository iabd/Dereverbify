import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, channelUp, channelDown):
        super(Attention, self).__init__()
        self.deconv = nn.ConvTranspose2d(channelDown, channelUp, kernel_size=1, stride=1)
        if channelUp / 2 == channelDown:
            self.conv1 = lambda a: a
        else:
            self.conv1 = nn.Conv2d(channelUp, channelUp, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(channelUp, 1, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        addition = self.deconv(g) + self.conv1(x)
        addition = self.relu(addition)
        addition = self.conv2(addition)
        attention = self.sigmoid(addition)
        attention = F.interpolate(addition.unsqueeze(0), size=x[0].shape, mode='trilinear').squeeze(0)
        return x.mul(attention)


class CBL(nn.Module):
    """Conv -> BN -> ReLU"""

    def __init__(self, inChannels, outChannels, midChannels=None):
        super(CBL, self).__init__()
        if not midChannels:
            midChannels = outChannels
        self.cbl = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=midChannels, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=midChannels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)

            # The original U-net paper used two layers of CBL

            # nn.Conv2d(in_channels=midChannels, out_channels=outChannels, kernel_size=5, padding=2),
            # nn.BatchNorm2d(num_features=outChannels),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, input_):
        return self.cbl(input_)


class Upsample(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels=inChannels, out_channels=inChannels // 2, kernel_size=5,
                                           stride=2)
        self.conv = CBL(inChannels, outChannels)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, inChannels, outChannels, maxpool=True):
        super(Downsample, self).__init__()
        self.isMaxpool = maxpool
        self.CBL = CBL(inChannels, outChannels)
        self.maxpool = nn.MaxPool2d(2)
        self.maxpoolConv = nn.Sequential(
            nn.MaxPool2d(2),
            CBL(inChannels, outChannels)
        )

    def forward(self, input_):
        if self.maxpool:
            return self.maxpool(self.CBL(input_))
        return self.CBL(input_)


class OutConv(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=1)
        self.sigm = nn.Sigmoid()

    def forward(self, input_):
        return self.sigm(self.conv(input_))


class UNet(nn.Module):
    def __init__(self, nChannels, nClasses, dropout=0.5):
        super(UNet, self).__init__()
        self.nChannels = nChannels
        self.nClasses = nClasses
        self.DCT = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, stride=2),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(dropout)
        self.inc = CBL(nChannels, 16)
        self.down1 = Downsample(16, 32)
        self.attention1=Attention(16, 32)
        self.down2 = Downsample(32, 64)
        self.attention2=Attention(32, 64)
        self.down3 = Downsample(64, 128)
        self.attention3=Attention(64, 128)
        self.down4 = Downsample(128, 256)
        self.attention4=Attention(128, 256)
        self.down5 = Downsample(256, 512)
        self.attention5=Attention(256, 512)
        self.up1 = Upsample(512, 256)
        self.up2 = Upsample(256, 128)
        self.up3 = Upsample(128, 64)
        self.up4 = Upsample(64, 32)
        self.up5 = Upsample(32, 16)
        self.up6 = OutConv(16, nClasses)

    def forward(self, x_):
        # C H W 1x256x256
        x1 = self.inc(x_) #16x256x256
        x2 = self.down1(x1) #32x128x128
        x3 = self.down2(x2) #64x64x64
        x4 = self.down3(x3) #128x32x32
        x5 = self.down4(x4) #256x16x16
        x6 = self.down5(x5) #512x8x8
        x = self.up1(x6, self.attention5(x5, x6)) #256
        x = self.up2(x, self.attention4(x4, x5)) #128
        x = self.up3(x, self.attention3(x3, x4)) #64
        x = self.up4(x, self.attention2(x2, x3)) #32
        x = self.up5(x, self.attention1(x1, x2)) #16
        output = self.up6(x) #1
        return output