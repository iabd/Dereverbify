import torch.nn as nn
from utils import DiceCoef, SSIM

class MixLoss:
    """

    Implements mixed loss of l1, l2, bce and dice. Accepts a dictionary as below:

        lossDict {
            "l1":0.2,
            "l2":0.3,
            "bce":0.4,
    "bceL":0.4,
            "dice":0.2,
    "ssim":1,
"nll":1
        }

    """
    def __init__(self, lossDict):
        assert type(lossDict) is dict, breakpoint()
        #assert list(lossDict.keys()) <= ['l1', 'l2', 'bce', 'bceL', 'dice', 'ssim', 'nll'], "lossDict should have keys that match ['l1', 'l2', 'bce', 'bceL', 'dice', 'ssim', 'nll]"

        self.lossDict=lossDict
        self.functions={
            "l1":nn.L1Loss(),
            "huber":nn.SmoothL1Loss(beta=0.1),
            "l2":nn.MSELoss(),
            "bce":nn.BCELoss(),
            "bceL":nn.BCEWithLogitsLoss(),
            "dice":DiceCoef(),
            "ssim":SSIM(),
            "nll":nn.NLLLoss()
        }


    def __call__(self, outputs, targets):
        overallLoss=0.0
        for loss, multiplier in self.lossDict.items():
            if multiplier !=0:
                overallLoss+=self.functions[loss](outputs, targets)*multiplier
        return overallLoss
