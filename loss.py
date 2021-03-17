import torch.nn as nn
from utils import DiceCoef, SSIM

class MixLoss:
    """

    Implements mixed loss of l1, l2, bce and dice. Accepts a dictionary as below:

        lossDict {
            "l1":0.2,
            "l2":0.3,
            "bce":0.4,
            "dice":0.2,
    "ssim":1
        }

    """
    def __init__(self, lossDict):
        assert type(lossDict) is dict, breakpoint()
        assert list(lossDict.keys()) <= ['l1', 'l2', 'bce', 'dice', 'ssim'], "lossDict should have keys that match ['l1', 'l2', 'bce', 'dice', 'ssim']"

        self.lossDict=lossDict
        self.functions={
            "l1":nn.L1Loss(),
            "l2":nn.MSELoss(),
            "bce":nn.BCELoss(),
            "dice":DiceCoef(),
            "ssim":SSIM(),
        }


    def __call__(self, outputs, targets):
        overallLoss=0.0
        for loss, multiplier in self.lossDict.items():
            if multiplier !=0:
                overallLoss+=self.functions[loss](outputs, targets)*multiplier
        return overallLoss
