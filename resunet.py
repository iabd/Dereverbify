import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Upsample(nn.Module):
    def __init__(self, inChannels, outChannels, kernel, padding, scale_factor=2):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = convrelu(inChannels, outChannels, kernel, padding)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return self.upsample(x)


class OutConv(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=1)
        self.sigm = nn.Sigmoid()

    def forward(self, input_):
        return self.sigm(self.conv(input_))


def convrelu(in_channels, out_channels, kernel=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


class ResUNet(nn.Module):
    def __init__(self, nChannels, nClasses, init_channels=16, dropout=0.3):
        super(ResUNet, self).__init__()
        self.nChannels = nChannels
        self.nClasses = nClasses
        teacher = models.resnet18(pretrained=True)
        self.teacher_layers = list(teacher.children())
        self.dropout = nn.Dropout(dropout)

        self.first_conv = self.teacher_layers[0]
        first_layer_weights = self.first_conv.state_dict()['weight']
        summed_weights = torch.sum(first_layer_weights, dim=1, keepdim=True)
        self.first_conv.weight.data = summed_weights
        self.first_conv.in_channels = 1

        self.down0 = convrelu(nChannels, 64, 3, 1)
        self.down0_ = convrelu(64, 64, 3, 1)

        self.down1 = nn.Sequential(*self.teacher_layers[1:3])
        self.down1_ = convrelu(64, 64, 1, 0)
        self.down2 = nn.Sequential(*self.teacher_layers[3:5])
        self.down2_ = convrelu(64, 64, 1, 0)
        self.down3 = nn.Sequential(*self.teacher_layers[5])
        self.down3_ = convrelu(128, 128, 1, 0)
        self.down4 = nn.Sequential(*self.teacher_layers[6])
        self.down4_ = convrelu(256, 256, 1, 0)
        self.down5 = nn.Sequential(*self.teacher_layers[7])
        self.down5_ = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up1 = Upsample(768, 512, 3, 1)
        self.up2 = Upsample(640, 256, 3, 1)
        self.up3 = Upsample(320, 256, 3, 1)
        self.up4 = Upsample(320, 128, 3, 1)

        self.final_layer = OutConv(192, nClasses)

    def forward(self, x):
        # C H W 1x256x256
        x_init = self.down0(x)
        x_init = self.down0_(x_init)
        x = self.first_conv(x)

        x1 = self.down1(x)  # 16x256x256
        x2 = self.down2(x1)  # 32x128x128
        x3 = self.down3(x2)  # 64x64x64
        x4 = self.down4(x3)  # 128x32x32
        x5 = self.down5(x4)  # 256x16x16

        x = self.down5_(x5)
        x = self.upsample(x5)
        x4 = self.down4_(x4)

        x = self.dropout(self.up1(x, x4))
        x3 = self.down3_(x3)
        x = self.dropout(self.up2(x, x3))
        x2 = self.down2_(x2)
        x = self.dropout(self.up3(x, x2))

        x1 = self.down1_(x1)
        x = self.up4(x, x1)

        x = torch.cat([x, x_init], dim=1)

        output = self.final_layer(x)  # 1
        return output



