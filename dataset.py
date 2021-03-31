from glob import glob
from os import listdir
import numpy as np
from itertools import chain
from sklearn import preprocessing
import librosa, torch, random
from torch.utils.data import IterableDataset, Dataset
from utils import polarToComplex, complexToPolar

class TrainDataset(IterableDataset):
    def __init__(self, wavPath, revPath, samplingRate, stftParams, shuffle=True, normalization="-1+1"):
        self.sr = samplingRate
        self.stftParams=stftParams
        self.wavPath = wavPath
        self.revPath = revPath
        self.normalization=normalization
        self.ids = [i for i in listdir(revPath) if not i.startswith('.')]
        if shuffle:
            random.shuffle(self.ids)

    def __len__(self):
        return len(self.ids)

    def transform(self, X):
        if self.normalization=="0+1":
            return (X-X.min())/(X.max()-X.min())
            #return preprocessing.normalize(X)
        elif self.normalization=="-1+1":
            X=2*(X-X.min())/(X.max()-X.min())
            return X-1
        else:
            raise NotImplementedError(self.__class__.transform)

    def squaredChunks(self, spec):
        n = self.stftParams['n_fft'] // 2
        l = spec.shape[1]
        for i in range(0, l - l % n, n):
            yield np.expand_dims(spec[:, i:i + n], axis=0)

    def getAudio(self, idx):
        org = glob(self.wavPath + idx)[0]
        rev = glob(self.revPath + idx)[0]
        org, _ = librosa.load(org, sr=self.sr)
        rev, _ = librosa.load(rev, sr=self.sr)
        org = librosa.power_to_db(np.abs(librosa.stft(org, **self.stftParams))[1:])
        rev = librosa.power_to_db(np.abs(librosa.stft(rev, **self.stftParams))[1:])
        orgArray = torch.FloatTensor(list(self.squaredChunks(self.transform(org))))
        revArray = torch.FloatTensor(list(self.squaredChunks(self.transform(rev))))[:orgArray.shape[0]]
        for i, v in enumerate(revArray):
            yield (orgArray[i], v)

    def getStream(self, ids):
        yield from chain.from_iterable(map(self.getAudio, ids))

    def __iter__(self):
        return self.getStream(self.ids)



class TestDataset(Dataset):
    def __init__(self, wavFile, samplingRate, stftParams):
        audio, _ = librosa.load(wavFile, sr=samplingRate)
        self.stft=librosa.stft(audio, **stftParams)
        self.istftParams={key:val for key, val in stftParams.items() if key!="n_fft"}
    
    def __len__(self):
        return self.stft.shape[0]
    
    def reconstructAudio(self, newMag):
        audio=[]
        for idx, batchM in enumerate(newMag):
            try:
                tempAudio=librosa.istft(polarToComplex(batchM.numpy()[0], self.phase[idx].numpy()[0]), **self.istftParams)
            except:
                tempAudio = librosa.istft(polarToComplex(batchM[0], self.phase[idx][0]), **self.istftParams)
                
            audio.extend(tempAudio)
        
        return audio
    
    
    def transform(self, X):
        """Normalizes the values between 0 and 1"""
        return (X - X.min()) / (X.max() - X.min())

    def squaredChunks(self, array):
        """Accepts numpy array of shape [x, y] returns squared arrays split along y-axis"""
        xdim=array.shape[0]//2*2
        ydim=array.shape[1]
        ydimHead=ydim//xdim*xdim #y-dim of the initial batches
        ydimTail=ydim-ydim//xdim*xdim # y-dim of the last batch 
        batches=np.asarray(np.hsplit(array[:xdim,:ydimHead], ydim//xdim))


        #pad the last batch with minimum value of spectrogram
        tailBatch=array[:xdim, ydimHead:]
        tailB=np.full((xdim, xdim), array.min())
        tailB[:, :ydimTail]=tailBatch
        batches=np.concatenate((batches, np.expand_dims(tailB, axis=0)), axis=0)
        batches=self.transform(batches)
        return np.expand_dims(batches, axis=1)
    
    def audioProcessing(self):
        mag, phase=complexToPolar(self.stft)
        mag=self.squaredChunks(mag)
        self.phase=self.squaredChunks(phase)
        return mag
    
    def __call__(self):
        return self.audioProcessing()
