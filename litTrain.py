import os, argparse, json

import torch
from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
from litModel import UNet

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration', default='config.json')
    args=parser.parse_args()
    with open(args.config) as f:
        data=f.read()
    config=json.loads(data)
    netParams=config["netParams"]
    dataConfig=config["dataConfig"]


    net=UNet(netParams)

    trainData=TrainDataset(**dataConfig['train'])
    valData=TrainDataset(**dataConfig['validation'])
    trainLoader=DataLoader(trainData, batch_size=config['batchSize'], num_workers=config['numWorkers'])
    valLoader=DataLoader(valData, batch_size=config['batchSize'], num_workers=config['numWorkers'])

    os.makedirs(config["logDir"], exist_ok=True)
    try:
        log_dir = sorted(os.listdir(config["logDir"]))[-1]
    except IndexError:
        log_dir = os.path.join(config["logDir"], 'version_0')
    checkpointCallback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        verbose=True,
    )
    stopCallback = EarlyStopping(
        monitor='val loss',
        mode='auto',
        patience=5,
        verbose=True,
    )


    trainer=Trainer(checkpoint_callback=checkpointCallback)


    trainer.fit(net, trainLoader, valLoader)
