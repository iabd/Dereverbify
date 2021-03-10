import argparse, json, torch, os, pdb
from dataset import TrainDataset
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
from unet import UNet
import torch.nn as nn
from tqdm import tqdm
import logging
from itertools import islice
from torch.utils.tensorboard import SummaryWriter
from preprocessAudio import tensorToImage, diceCoef



def validate(net, valLoader, device):
    tot=0
    
    with tqdm(total=500, desc='Validation round', unit='batch', leave=False) as pbar:
        for idx, batch in enumerate(islice(valLoader, 100)):
            
            revSpecs=batch[1].to(device=device, dtype=torch.float32)
            orgSpecs=batch[0].to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred=net(revSpecs)

            tot+=F.binary_cross_entropy(pred, orgSpecs)

            pbar.update()

    net.train()
    return tot/len(valLoader)


    return 0

def train(batchSize,lr, epochs, device, saveEvery, checkpointPath, finetune, bceWeight,dataConfig, **valConfig):
    writer=SummaryWriter()
    trainData=TrainDataset(**dataConfig)
    valData=TrainDataset(**valConfig)
    trainLoader=DataLoader(trainData, batch_size=batchSize, shuffle=False, num_workers=4)
    valLoader=DataLoader(valData, batch_size=batchSize, shuffle=False, num_workers=4)

    net=UNet(1, 1)
    if not finetune:
        net.cuda()
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCELoss()
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    globalStep=0
    if finetune:
        print("LOADING CHECKPOINT __")
        checkpoint=torch.load("checkpoint.pt", map_location='cpu')
        net.load_state_dict(checkpoint['modelStateDict'])
        net.cuda()
        optimizer.load_state_dict(checkpoint['optimizerStateDict'])
        epoch=checkpoint['epoch']
        loss=checkpoint['loss']

        
    for epoch in range(epochs):
        net.train()
        epochLoss=0

        with tqdm(total=epochs, desc="Epoch {}/{}".format(epoch+1, epochs), unit="audio", leave=False) as pbar:
            for idx, batch in enumerate(islice(trainLoader, 2002)):
                orgSpecs = batch[0].to(device=device, dtype=torch.float32)
                revdSpecs=batch[1].to(device=device, dtype=torch.float32)
                genSpecs=net(revdSpecs)
                bceLoss=criterion(genSpecs, orgSpecs)
                diceLoss=diceCoef(genSpecs, orgSpecs)
                loss=bceLoss*bceWeight+diceLoss*(1-bceWeight)
                epochLoss+=loss.item()
                writer.add_scalar('train loss', loss.item(), globalStep)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                globalStep+=1

                if idx%saveEvery==0:
                    print("saving model ..")
                    torch.save({
                        'epoch': epoch,
                        'modelStateDict': net.state_dict(),
                        'optimizerStateDict': optimizer.state_dict(),
                        'loss': loss,
                    }, 'checkpoint.pt')

                
                    for tag, value in net.named_parameters():
                        tag=tag.replace(".", "/")
                        writer.add_histogram('weights/'+tag, value.data.cpu().numpy(), globalStep)
                        if value.grad is not None:
                            writer.add_histogram('grads/'+tag, value.grad.data.cpu().numpy(), globalStep)

                    valScore=validate(net, valLoader, device)
                    scheduler.step(valScore)
                    writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], globalStep)
                    logging.info('Validation cross entropy: {}'.format(valScore))
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
    train(dataConfig=dataConfig, **trainConfig, **valConfig)
