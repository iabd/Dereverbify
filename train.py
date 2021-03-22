import argparse, json, torch, pdb, librosa, os
from glob import glob
import numpy as np
from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import logging
from itertools import islice
from torch.utils.tensorboard import SummaryWriter
from loss import MixLoss
from utils import countParams

def testIterations(net, path, samplingRate, batchSize, device, stftParams):
   # model.to(device)
    net.eval()
    generatedAudios=[]
    filelist=[i for i in os.listdir(path) if not i.startswith('.')]
    for idx, file in enumerate(filelist):
        wavpath=glob(path+file)[0]
        dset=TestDataset(wavpath, samplingRate, stftParams)
        inp=torch.from_numpy(dset()).to(device=device, dtype=torch.float32)
        datapoints=inp.shape[0]
        for i in range(0, datapoints, batchSize):
            print("\nGenerating Audio {}/{}:  Progress... {}/{}".format(idx+1, len(filelist), i, datapoints))
            with torch.no_grad():
                if i==0:
                    output=net(inp[i:i+batchSize])
                else:
                    output=torch.cat((output, inp[i:i+batchSize]))
                    
        generatedAudios.append(dset.reconstructAudio(output.cpu().detach().numpy()))
    net.train()
    return generatedAudios


def validate(net, valLoader, device, valCriterion):
    tot=0
    with tqdm(total=100, desc='Validation round', unit='batch', leave=False) as pbar:
        for idx, batch in enumerate(valLoader):
            
            revSpecs=batch[1].to(device=device, dtype=torch.float32)
            orgSpecs=batch[0].to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred=net(revSpecs)
            tot+=valCriterion(pred, orgSpecs)

            pbar.update()
            if idx==100:
                break
    
    net.train()
    return tot/len(valLoader)


def train(batchSize,lr, epochs, device, saveEvery, checkpointPath, finetune, unetType,dataConfig, trainLossConfig, valLossConfig, testParams):
    writer=SummaryWriter()
    trainData=TrainDataset(**dataConfig['train'])
    originalSound, _=librosa.load('LJ050-0084.wav', sr=16000)
    valData=TrainDataset(**dataConfig['validation'])
    trainLoader=DataLoader(trainData, batch_size=batchSize,  num_workers=4)
    valLoader=DataLoader(valData, batch_size=batchSize, num_workers=4)
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
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    globalStep=0
    if finetune:
        print("LOADING CHECKPOINT __")
        checkpoint=torch.load(checkpointPath, map_location='cpu')
        net.load_state_dict(checkpoint['modelStateDict'])
        net.cuda()
        #optimizer.load_state_dict(checkpoint['optimizerStateDict'])


    params=countParams(net)
    print("Initializing training with {} params.".format(params))
        
    for epoch in range(epochs):
        net.train()
        epochLoss=0

        with tqdm(total=epochs, desc="Epoch {}/{}".format(epoch+1, epochs), unit="audio", leave=False) as pbar:
            for idx, batch in enumerate(trainLoader):
                orgSpecs = batch[0].to(device=device, dtype=torch.float32)
                revdSpecs=batch[1].to(device=device, dtype=torch.float32)
                genSpecs=net(revdSpecs)

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
                    }, 'newExp{}Checkpoint.pt'.format(unetType))
                    
                
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
                    writer.add_audio('Original Audio', originalSound, global_step=globalStep, sample_rate=16000)
                    print("Generating audios .. ")
                    genAudios=testIterations(net, **testParams)
                    
                    for ii, aud in enumerate(genAudios):
                        writer.add_audio('Generated Audio {}'.format(ii), np.asarray(aud), global_step=globalStep, sample_rate=1600)


                    ## TODO : Add spectrogram images to writer

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
    train(**trainConfig, dataConfig=dataConfig, trainLossConfig=config["trainLossConfig"], valLossConfig=config["valLossConfig"], testParams=config["testParams"])
