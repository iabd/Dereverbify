import argparse, json, torch, os
from dataset import TrainDataset
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
from unet import UNet
import torch.nn as nn
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter




def validate(net, valLoader, device):
    tot=0
    with tqdm(n_val=len(valLoader), desc='Validation round', unit='batch', leave=False) as pbar:
        for idx, batch in enumerate(valLoader):
            revSpecs=batch[1].to(device=device, dtype=torch.float32)
            orgSpecs=batch[0].to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred=net(revSpecs)

            tot+=F.cross_entropy(pred, orgSpecs)

            pbar.update()

    net.train()
    return tot/=len(valLoader)


    return 0
def train(batchSize,lr, epochs, device, saveEvery, checkpointPath, **dataConfig):
    writer=SummaryWriter()
    dataset=TrainDataset(**dataConfig)
    nVal=int(len(dataset)*0.2)
    nTrain=len(dataset)-nVal
    trainData, valData=random_split(dataset, [nTrain, nVal])
    trainLoader=DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=4)
    valLoader=DataLoader(valData, batch_size=batchSize, shuffle=True, num_workers=4)


    net=UNet(1, 1)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    globalStep=0
    if os.path.exists(checkpointPath):
        print("LOADING CHECKPOINT __")
        checkpoint=torch.load("checkpoint.pt")
        net.load_state_dict(checkpoint['modelStateDict'])
        optimizer.load_state_dict(checkpoint['optimizerStateDict'])
        epoch=checkpoint['epoch']
        loss=checkpoint['loss']

    for epoch in range(epochs):
        net.train()
        epochLoss=0

        with tqdm(total=epochs, desc="Epoch {}/{}".format(epoch+1, epochs), unit="img") as pbar:
            for idx, batch in enumerate(trainLoader):
                orgSpecs = batch[0].to(device=device, dtype=torch.float32)
                revdSpecs=batch[1].to(device=device, dtype=torch.float32)
                genSpecs=net(revdSpecs)
                loss=criterion(genSpecs, orgSpecs)
                epochLoss+=loss.item()
                writer.add_scalar('train loss', loss.item(), globalStep)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                globalStep+=1

                if idx==saveEvery:
                    print("saving model ..")
                    torch.save({
                        'epoch': epoch,
                        'modelStateDict': net.state_dict(),
                        'optimizerStateDict': optimizer.state_dict(),
                        'loss': loss,
                    }, 'checkpoint.pt')

                if globalStep % (epochs // (10*batchSize))==0:
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
                    writer.add_images('Target Specs', orgSpecs[0], globalStep)
                    writer.add_images('Generated Specs', genSpecs[0], globalStep)

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, 'checkpoint.pt')

    writer.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')
    args=parser.parse_args()
    with open(args.config) as f:
        data=f.read()
    config=json.loads(data)
    trainConfig=config["trainConfig"]
    # dataConfig=config["dataConfig"]
    numGPUs=torch.cuda.device_count()
    train(**trainConfig)
