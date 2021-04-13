import matplotlib, os
matplotlib.use('Agg')
import librosa
from matplotlib import cm
import numpy as np
import pylab, torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from librosa import display
from PIL import Image
from skimage import metrics


def getFileList(dirName):
    listOfFile = [i for i in os.listdir(dirName) if not i.startswith(".")]
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getFileList(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def reconstructAudioFromBatches(magB, phaseB, istftParams=None):
    audio=[]
    for idx, batchM in enumerate(magB):
        try:
            tempAudio=librosa.istft(polarToComplex(batchM.numpy()[0], phaseB[idx].numpy()[0]), **istftParams)
        except:
            tempAudio = librosa.istft(polarToComplex(batchM[0], phaseB[idx][0]), **istftParams)


        audio.extend(tempAudio)
    return audio

def complexToPolar(com):
    mag=np.abs(com)
    phase=np.angle(com, deg=False) # in radians
    return mag, phase
def polarToComplex(mag, phase):
    imag=np.cos(phase)+1j*np.sin(phase)
    return mag*imag

class SSIM():
    def __call__(self, prediction, target):
        ssims=0.
        p=prediction.cpu().detach().numpy()
        t=target.cpu().detach().numpy()
        for idx, batch in enumerate(p):
            range_=batch.max()-batch.min()
            ssims+=metrics.structural_similarity(batch[0], t[idx][0], data_range=range_)
        return 1-(ssims/(idx+1))

def countParams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

class DiceCoef():
    def __init__(self, smooth=1.):
        self.smooth=smooth

    def __call__(self, prediction, target):
        p=prediction.contiguous()
        t=target.contiguous()

        intersection=(p*t).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) / (p.sum(dim=2).sum(dim=2) + t.sum(dim=2).sum(dim=2) + self.smooth)))
        return loss.mean()

#     def __call__(self, pr):
# def dice(prediction, target, smooth=1.):
#     p=prediction.contiguous()
#     t=target.contiguous()
#
#     intersection=(p*t).sum(dim=2).sum(dim=2)
#     loss=(1-((2.*intersection+smooth)/(p.sum(dim=2).sum(dim=2)+t.sum(dim=2).sum(dim=2)+smooth)))
#     return loss.mean()
    
def fetchProgress(dir, ids):
    try:
        with open(dir+"progress", 'r') as f:
            p=f.readline()
            return ids.index(p)+1
    except:
        return 0

def tensorToImage(tensor):
    fig=plt.Figure()
    canvas=FigureCanvas(fig)
    ax=fig.add_subplot(111)
    p=librosa.display.specshow(librosa.amplitude_to_db(tensor.cpu().detach().numpy()**2), ax=ax, y_axis='hz', x_axis='time', cmap=cm.jet)

    pylab.savefig("writer.jpg", bbox_inches=None, pad_inches=0)
    img=Image.open("writer.jpg").convert('RGB')
    img=torch.from_numpy(np.array(img))
    return img.unsqueeze(0).permute(0, 3, 1,2)


    
def clipRevAudio(org, rev):
    nonzeroOrg=np.nonzero(org)[0][0]
    nonzeroRev=np.nonzero(rev)[0][0]
    if nonzeroOrg<nonzeroRev:
        rev=rev[nonzeroRev:]

    clipBar=org[:100].mean()
    clipBar=np.argmax(rev>clipBar)
    return rev[clipBar:]


def saveSpectrogram(s1, s2, name='tensorboardImage.jpg'):
    pylab.figure(figsize=(8,3))
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge                                                                                        
    plt.subplot(1, 2, 1)
    librosa.display.specshow(librosa.power_to_db(s1), y_axis='hz', x_axis='time', cmap=cm.jet)
    plt.subplot(1, 2, 2)
    librosa.display.specshow(librosa.power_to_db(s2), y_axis='hz', x_axis='time', cmap=cm.jet)
    pylab.savefig(name, bbox_inches=None, pad_inches=0)
    pylab.close()
    return name





