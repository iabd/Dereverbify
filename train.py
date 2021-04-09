import argparse, json, torch, os
from dataset import TrainDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.notebook import tqdm
import logging
from utils import saveSpectrogram
from torchvision import transforms
from PIL import Image
from unet import UNet
from torch.utils.tensorboard import SummaryWriter
from loss import MixLoss
from utils import countParams

def validate(net, valLoader, device, valCriterion):
    tot=0
    with tqdm(total=100, desc='Validation round', unit='batch', position=3, leave=True) as pbar:
        for idx, batch in enumerate(valLoader):
            
            revSpecs=batch[1].to(device=device, dtype=torch.float32)
            orgSpecs=batch[0].to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred=net(revSpecs)
            tot+=valCriterion(pred, orgSpecs)

            pbar.update()
            if idx==100:
                saveSpectrogram(revSpecs[0][0].cpu().numpy(), pred[0][0].cpu().numpy())
                break
    
    net.train()
    return tot/100


def train(driveDir, batchSize,lr, epochs, device, saveEvery, checkpointPath, finetune,dataConfig, trainLossConfig, valLossConfig):
    writer=SummaryWriter(os.path.join(driveDir, "runs"))
    trainData=TrainDataset(**dataConfig['train'])
    valData=TrainDataset(**dataConfig['validation'])
    trainLoader=DataLoader(trainData, batch_size=batchSize,  num_workers=2)
    valLoader=DataLoader(valData, batch_size=batchSize, num_workers=2)

    net=UNet(1, 1)

    if not finetune and device=="cuda":
        net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
    trainCriterion=MixLoss(trainLossConfig)
    valCriterion=MixLoss(valLossConfig)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    globalStep=0
    if finetune:
        print("LOADING CHECKPOINT __")
        checkpoint=torch.load(checkpointPath, map_location='cpu')
        net.load_state_dict(checkpoint['modelStateDict'])
        net.cuda()
        optimizer.load_state_dict(checkpoint['optimizerStateDict'])

    epoch=0
    params=countParams(net)
    print("Initializing training with {} params.".format(params))
    with tqdm(total=epochs, desc="Epoch {}/{}".format(epoch + 1, epochs), unit="audio", position=0,leave=True) as pbar:
        for epoch in range(epochs):
            net.train()
            epochLoss=0
            with tqdm(total=len(trainLoader), desc="training iterations", unit="batch", position=1, leave=True) as pbar2:
                for idx, batch in enumerate(trainLoader):
                    if idx==12001:
                        break
                    orgSpecs = batch[0].to(device=device, dtype=torch.float32)
                    revdSpecs=batch[1].to(device=device, dtype=torch.float32)
                    genSpecs=net(revdSpecs)

                    loss=trainCriterion(genSpecs, orgSpecs)
                    epochLoss+=loss.item()
                    writer.add_scalar('train loss', loss.item(), globalStep)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    optimizer.zero_grad()
                    loss.backward()
                    #nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()
                    globalStep+=1

                    if (idx+1)%saveEvery==0:
                        print("saving model ..")
                        torch.save({
                            'epoch': epoch,
                            'modelStateDict': net.state_dict(),
                            'optimizerStateDict': optimizer.state_dict(),
                            'loss': loss,
                        }, os.path.join(driveDir, 'colabBatchSize1_{}.pt'.format(idx+1)))

                    if (idx+1)%500==0:
                        print("saving model ..")
                        torch.save({
                            'epoch': epoch,
                            'modelStateDict': net.state_dict(),
                            'optimizerStateDict': optimizer.state_dict(),
                            'loss': loss,
                            'iteration':idx
                        }, os.path.join(driveDir, '.colabBatchSize1'.format(idx)))
                        for tag, value in net.named_parameters():
                            tag=tag.replace(".", "/")
                            writer.add_histogram('weights/'+tag, value.data.cpu().numpy(), globalStep)
                            if value.grad is not None:
                                writer.add_histogram('grads/'+tag, value.grad.data.cpu().numpy(), globalStep)

                        valScore=validate(net, valLoader, device, valCriterion)
                        scheduler.step(valScore)
                        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], globalStep)

                        img_=transforms.ToTensor()(Image.open('tensorboardImage.jpg'))
                        writer.add_image('Spectrogram', img_, global_step=globalStep)

                        logging.info('Validation Score: {}'.format(valScore))
                        writer.add_scalar('Dice/test', valScore, globalStep)
                        #writer.add_audio('Original Audio', originalSound, global_step=globalStep, sample_rate=16000)


                    pbar2.update()

        pbar.update()
    writer.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')
    args=parser.parse_args()
    with open(args.config) as f:
        data=f.read()
    config=json.loads(data)
    trainConfig=config["trainConfig"]
    dataConfig=config["dataConfig"]
    train(**trainConfig, dataConfig=dataConfig, trainLossConfig=config["trainLossConfig"], valLossConfig=config["valLossConfig"])
