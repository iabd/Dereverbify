import argparse, json, torch, pdb
from dataset import TrainDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import logging
from itertools import islice
from torch.utils.tensorboard import SummaryWriter
from loss import MixLoss
from utils import countParams

def validate(net, valLoader, device, valCriterion):
    tot=0
    with tqdm(total=100, desc='Validation round', unit='batch', leave=False) as pbar:
        for idx, batch in enumerate(islice(valLoader, 100)):
            
            revSpecs=batch[1].to(device=device, dtype=torch.float32)
            orgSpecs=batch[0].to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred=net(revSpecs)
            tot+=valCriterion(pred, orgSpecs)

            pbar.update()

    net.train()
    return tot/len(valLoader)


def train(batchSize,lr, epochs, device, saveEvery, checkpointPath, finetune, unetType,dataConfig, valConfig, trainLossConfig, valLossConfig):
    writer=SummaryWriter()
    trainData=TrainDataset(**dataConfig)
    valData=TrainDataset(**valConfig)
    trainLoader=DataLoader(trainData, batch_size=batchSize, shuffle=False, num_workers=4)
    valLoader=DataLoader(valData, batch_size=batchSize, shuffle=False, num_workers=4)
    if unetType=="small":
        from unet import UNet
        net=UNet(1, 1)
    else:
        from unet2 import UNet
        net=UNet(1,1)
    if not finetune and device=="cuda":
        net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    trainCriterion=MixLoss(trainLossConfig)
    valCriterion=MixLoss(valLossConfig)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    globalStep=0
    if finetune:
        print("LOADING CHECKPOINT __")
        checkpoint=torch.load(checkpointPath, map_location='cpu')
        net.load_state_dict(checkpoint['modelStateDict'])
        net.cuda()
        optimizer.load_state_dict(checkpoint['optimizerStateDict'])


    params=countParams(net)
    print("Initializing training with {} params.".format(params))
        
    for epoch in range(epochs):
        net.train()
        epochLoss=0

        with tqdm(total=epochs, desc="Epoch {}/{}".format(epoch+1, epochs), unit="audio", leave=False) as pbar:
            for idx, batch in enumerate(islice(trainLoader, 2011)):
                orgSpecs = batch[0].to(device=device, dtype=torch.float32)
                revdSpecs=batch[1].to(device=device, dtype=torch.float32)
                genSpecs=net(revdSpecs)
                # mseLoss=criterion(genSpecs, orgSpecs)
                # diceLoss=diceCoef(genSpecs, orgSpecs)
                # loss=mseLoss*bceWeight+diceLoss*(1-bceWeight)
                loss=trainCriterion(genSpecs, orgSpecs)
                epochLoss+=loss.item()
                writer.add_scalar('train loss', loss.item(), globalStep)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                globalStep+=1

                if (idx+1)%saveEvery==0:
                    print("saving model ..")
                    torch.save({
                        'epoch': epoch,
                        'modelStateDict': net.state_dict(),
                        'optimizerStateDict': optimizer.state_dict(),
                        'loss': loss,
                    }, '{}Checkpoint.pt'.format(unetType))
                    
                
                    for tag, value in net.named_parameters():
                        tag=tag.replace(".", "/")
                        writer.add_histogram('weights/'+tag, value.data.cpu().numpy(), globalStep)
                        if value.grad is not None:
                            writer.add_histogram('grads/'+tag, value.grad.data.cpu().numpy(), globalStep)

                    valScore=validate(net, valLoader, device, valCriterion)
                    scheduler.step(valScore)
                    writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], globalStep)
                    logging.info('Validation Score: {}'.format(valScore))
                    writer.add_scalar('Dice/test', valScore, globalStep)
                    #image=tensorToImage(orgSpecs[0][0])
                    #writer.add_images('Target Specs', image, globalStep)
                    #writer.add_images('Generated Specs', tensorToImage(genSpecs[0][0]), globalStep)

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
    valConfig=config["valConfig"]
    
    numGPUs=torch.cuda.device_count()
    train(**trainConfig, dataConfig=dataConfig,  valConfig=valConfig, trainLossConfig=config["trainLossConfig"], valLossConfig=config["valLossConfig"])
