import torch
import torch.nn as nn
import torch.nn.functional as F

class CBL(nn.Module):
    """Conv -> BN -> ReLU"""

    def __init__(self, inChannels, outChannels, midChannels=None):
        super(CBL, self).__init__()
        if not midChannels:
            midChannels=outChannels
        self.cbl=nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=midChannels, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=midChannels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)

            #The original U-net paper used two layers of CBL

            # nn.Conv2d(in_channels=midChannels, out_channels=outChannels, kernel_size=5, padding=2),
            # nn.BatchNorm2d(num_features=outChannels),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, input_):
        return self.cbl(input_)





class Upsample(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(Upsample, self).__init__()

        self.upsample=nn.ConvTranspose2d(in_channels=inChannels, out_channels=inChannels//2, kernel_size=5, stride=2)
        self.conv=CBL(inChannels, outChannels)

    def forward(self, x1, x2):
        x1=self.upsample(x1)
        diffY=x2.size()[2]-x1.size()[2]
        diffX=x2.size()[3]-x1.size()[3]

        x1=F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x=torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, inChannels, outChannels, maxpool=True):
        super(Downsample, self).__init__()
        self.isMaxpool=maxpool
        self.CBL=CBL(inChannels, outChannels)
        self.maxpool=nn.MaxPool2d(2)
        self.maxpoolConv=nn.Sequential(
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
        self.conv=nn.Conv2d(inChannels, outChannels, kernel_size=1)
        self.sigm=nn.Sigmoid()

    def forward(self, input_):
        return self.sigm(self.conv(input_))



class UNet(nn.Module):
    def __init__(self, nChannels, nClasses):
        super(UNet, self).__init__()
        self.nChannels=nChannels
        self.nClasses=nClasses
        self.DCT=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, stride=2),
            nn.Tanh()
        )
        self.inc=CBL(nChannels, 64)
        self.down1=Downsample(64, 128)
        self.down2=Downsample(128, 256)
        self.down3=Downsample(256, 512)
        self.down4=Downsample(512, 512)
        self.down5=Downsample(512, 512)
        self.down6=Downsample(512, 512)
        self.down7=Downsample(512, 1024)
        
        self.up1=Upsample(1024, 1024)
        self.up2=Upsample(1024, 1024)
        self.dropout=nn.Dropout(0.2)
        self.up3=Upsample(1024, 1024)
        self.up4=Upsample(1024, 512)
        self.up5=Upsample(512, 256)
        self.up6=Upsample(256, 128)
        self.up7=Upsample(128, 64)
        self.outc=OutConv(64, nClasses)

    def forward(self, x):
        x1=self.inc(x) # 128x128x64
        x2=self.down1(x1) # 64x64x128
        x3=self.down2(x2) # 32x32x256
        x4=self.down3(x3) # 16x16x512
        x5=self.down4(x4) # 8x8x512
        x6=self.down5(x5) # 4x4x512
        x7=self.down6(x6) # 2x2x512
        x8=self.down7(x7) # 1x1x1024
        x=self.up1(x8, x7) # 2x2x1024
        x=self.up2(x, x6) # 4x4x1024
        x=self.dropout(x)
        x=self.up3(x, x5) # 8x8x1024
        x=self.up4(x, x4)
        x=self.dropout(x)
        # 16x16x512
        x=self.up5(x, x3) # 32x32x256
        x = self.up6(x, x2) # 64x64x128
        x=self.dropout(x)
        x = self.up7(x, x1) # 128x128x128
        logits=self.outc(x)# 1x128x128
        return logits
