from glob import glob
from os import listdir
import numpy as np
from itertools import chain, cycle
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
import librosa, torch, random


class TrainDataset(IterableDataset):
    def __init__(self, wavPath, revPath, samplingRate, segmentLength, nfft, winLength, window, shuffle=True):
        self.sr = samplingRate
        self.segmentLength = segmentLength
        self.nfft = nfft
        self.window = window
        self.winLength = winLength
        self.wavPath = wavPath
        self.revPath = revPath
        self.ids = [i for i in listdir(revPath) if not i.startswith('.')]

        if shuffle:
            random.shuffle(self.ids)

    def __len__(self):
        return len(self.ids)

    def transform(self, X):
        return (X-X.min())/(X.max()-X.min())
    
    def squaredChunks(self, spec, n=256):
        l = len(spec)
        for i in range(0, l - l % n, n):
            yield np.expand_dims(spec[i:i + n].T, axis=0)

    def getAudio(self, idx):
        org = glob(self.wavPath + idx)[0]
        rev = glob(self.revPath + idx)[0]
        org, _ = librosa.load(org, sr=self.sr)
        rev, _ = librosa.load(rev, sr=self.sr)
        org = np.abs(librosa.stft(org, n_fft=self.nfft, window=self.window, win_length=self.winLength))[1:,
              :self.segmentLength]
        rev = np.abs(librosa.stft(rev, n_fft=self.nfft, window=self.window, win_length=self.winLength))[1:,
              :self.segmentLength]
        orgArray = self.transform(torch.FloatTensor(list(self.squaredChunks(np.abs(org.T)))))
        revArray = self.transform(torch.FloatTensor(list(self.squaredChunks(np.abs(rev.T)))))
        for i, v in enumerate(revArray):
            yield (orgArray[i],v)

    def getStream(self, ids):
        yield from chain.from_iterable(map(self.getAudio, cycle(ids)))

    def __iter__(self):
        return self.getStream(self.ids)
