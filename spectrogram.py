import os, logging, librosa
from glob import glob
import numpy as np
from preprocessAudio import saveSpectrogram, fetchProgress
from tqdm import tqdm



class Spectrogram:
    def __init__(self, orgDir, revDir, specDir, sr=16000):
        self.orgDir=orgDir
        self.revDir=revDir
        self.specDir=specDir
        self.sr=sr
        self.ids=sorted([os.path.splitext(file)[0] for file in os.listdir(orgDir) if not file.startswith('.')])
        self.currentProgress=fetchProgress(self.specDir, self.ids)
        logging.info("Creating dataset with {} examples".format(len(self.ids)))

    @classmethod
    def getSpectrogram(cls, a1, a2):
        a1 = librosa.stft(a1, n_fft=512, window='hann', win_length=32)
        a2 = librosa.stft(a2, n_fft=512, window='hann', win_length=32)
        # breakpoint()
        a2 = a2[:, :np.where(abs(a2) == 0)[1][0]]  # clip from zeros which will get us -inf values for log
        a1 = librosa.power_to_db(np.abs(a1)**2)
        ##padding a1 to be the same size as a2
        # temp=np.zeros(a2.shape)
        # temp[:a1.shape[0], :a1.shape[1]]=a1
        # a1=temp
        # temp=0

        a2Phase=a2.imag
        a2=librosa.power_to_db(np.abs(a2)**2)
        return a1, a2, a2Phase



    def go(self):
        with tqdm(total=len(self.ids[self.currentProgress:]), position=self.currentProgress, leave=True) as pbar:
            for idx in self.ids[self.currentProgress:]:
                orgAudio=glob(self.orgDir+idx+".wav")
                revAudio=glob(self.revDir+idx+".wav")
                orgAudio, _=librosa.load(orgAudio[0], sr=self.sr)
                revAudio, _=librosa.load(revAudio[0], sr=self.sr)

                org, rev, phase=self.getSpectrogram(orgAudio, revAudio)

                for i in range(org.shape[1]):
                    tempOrg=org[:, i*256:(i+1)*256]
                    # breakpoint()
                    if tempOrg.shape[1]==256:
                        saveSpectrogram(tempOrg, name=self.specDir+"org_"+idx+"_{}".format(i))
                        saveSpectrogram(rev[:, i*256:(i+1)*256], name=self.specDir+"rev_"+idx+"_{}".format(i))
                        np.save(self.specDir+idx+"_{}".format(i)+".npy", phase[:, i*256:(i+1)*256])

                with open(self.specDir+"progress", 'w') as f:
                    f.write(idx)

                pbar.update()

