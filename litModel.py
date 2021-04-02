import pytorch_lightning as pl
import torch.nn.functional as F
import torch, pdb
import torch.optim as optim
import torch.nn as nn


class UNet(pl.LightningModule):
    def __init__(self, params):
        super(UNet, self).__init__()
        self.lr=params["lr"]
        def cbl(inChannels, outChannels):
            return nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=5, padding=2),
                nn.BatchNorm2d(outChannels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        def downsample(inChannels, outChannels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                cbl(inChannels, outChannels)
            )

        class Upsample(nn.Module):
            def __init__(self, inChannels, outChannels):
                super(Upsample, self).__init__()
                self.upsample=nn.ConvTranspose2d(inChannels, outChannels, kernel_size=5, stride=2)
                self.conv=cbl(inChannels, outChannels)

            def forward(self, x1, x2):
                x1=self.upsample(x1)
                diffY=x2.size()[2]-x1.size()[2]
                diffX=x2.size()[3]-x1.size()[3]
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)

        def outConv(inChannels, outChannels):
            return nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1),
                nn.Sigmoid()
            )


        self.dropout=nn.Dropout(params["dropout"])
        self.inc=cbl(params["inChannels"], 16)
        self.down1=downsample(16, 32)
        self.down2=downsample(32, 64)
        self.down3=downsample(64, 128)
        self.down4=downsample(128, 256)
        self.down5=downsample(256, 512)

        self.up1=Upsample(512, 256)
        self.up2=Upsample(256, 128)
        self.up3=Upsample(128, 64)
        self.up4=Upsample(64, 32)
        self.up5=Upsample(32, 16)
        self.up6=outConv(16, params["outChannels"])

    def forward(self, x):
        x1=self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.dropout(x)
        x = self.up2(x, x4)
        x = self.dropout(x)
        x = self.up3(x, x3)
        x = self.dropout(x)
        x = self.up4(x, x2)
        x = self.dropout(x)
        x = self.up5(x, x1)
        output = self.up6(x)
        return output



    def configure_optimizers(self):
        optimizer=optim.Adam(self.parameters(), lr=self.lr)
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        # return [optimizer], [scheduler]
        return {
            'optimizer':optimizer,
            'lr_scheduler':scheduler,
            'monitor':'val loss'
        }
    def training_step(self, trainBatch, batch_idx):
        orgSpecs=trainBatch[0]
        revdSpecs=trainBatch[1]

        genSpecs=self.forward(revdSpecs)
        loss=F.binary_cross_entropy(genSpecs, orgSpecs)

        self.log("train loss", loss)
        return loss

    def validation_step(self, valBatch, batch_idx):
        orgSpecs=valBatch[0]
        revdSpecs=valBatch[1]
        genSpecs=self.forward(revdSpecs)
        loss=F.binary_cross_entropy(genSpecs, orgSpecs)
        return loss

    def validation_end(self, valOutputs):
        valLoss=valOutputs.mean()
        self.log("val loss", valLoss)


