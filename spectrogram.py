import os, librosa
from glob import glob
import numpy as np
from utils import saveSpectrogram, fetchProgress
from tqdm import tqdm



class Spectrogram:
    def __init__(self, orgDir, revDir, specDir, sr=16000):
        self.orgDir=orgDir
        self.revDir=revDir
        self.specDir=specDir
        self.sr=sr
        self.ids=sorted([os.path.splitext(file)[0] for file in os.listdir(revDir) if not file.startswith('.')])
        self.currentProgress=fetchProgress(self.specDir, self.ids)


    def __len__(self):
        return len(self.ids)


    @classmethod
    def polarToComplex(cls, mag, phase):
        imag=np.cos(phase)+1j*np.sin(phase)
        return mag*imag

    @classmethod
    def complexToPolar(cls, stftMatrix):
        mag=np.abs(stftMatrix)
        phase=np.angle(stftMatrix, deg=True)
        return mag, phase


    def polarToAudio(self, mag, phase):

        if not mag.shape==phase.shape:
            phase=phase[:mag.shape[0], :mag.shape[1]]

        spectrum=self.polarToComplex(mag, phase)
        audio=librosa.istft(spectrum, win_length=32, window='hamming')
        return audio


    def audiosToPolar(self, a1, a2):
        a1 = librosa.stft(a1, n_fft=512, window='hamming', win_length=32)
        a2 = librosa.stft(a2, n_fft=512, window='hamming', win_length=32)
        try:
            a2 = a2[:, :np.where(abs(a2) == 0)[1][0]]  # clip from zeros which will get us -inf values for log
        except:
            pass

        magA1, _=self.complexToPolar(a1)
        magA2, phase=self.complexToPolar(a2)
        return magA1, magA2, phase



    def lwSpectrum(self):
        for idx in tqdm(self.ids[self.currentProgress:]):
            orgAudio = glob(self.orgDir + idx + ".wav")
            revAudio = glob(self.revDir + idx + ".wav")
            orgAudio, _ = librosa.load(orgAudio[0], sr=self.sr)
            revAudio, _ = librosa.load(revAudio[0], sr=self.sr)

            org, rev, phase=self.audiosToPolar(orgAudio, revAudio)

            for i in range(org.shape[1]):
                tempOrg=org[:, i*256:(i+1)*256]
                if tempOrg.shape[1]==256:
                    np.savetxt(self.specDir+"org_"+idx+"{}".format(i)+".npy", org)
                    np.savetxt(self.specDir+"rev_"+idx+"{}".format(i)+".npy", rev)
                    np.savetxt(self.specDir + "phase_" + idx + "{}".format(i)+".npy", phase)

            with open(self.specDir+"progress", "w") as f:
                f.write(idx)

    def lwhSpectrum(self):
        with tqdm(total=len(self.ids[self.currentProgress:]), position=self.currentProgress, leave=True) as pbar:
            for idx in self.ids[self.currentProgress:]:
                orgAudio=glob(self.orgDir+idx+".wav")
                revAudio=glob(self.revDir+idx+".wav")
                orgAudio, _=librosa.load(orgAudio[0], sr=self.sr)
                revAudio, _=librosa.load(revAudio[0], sr=self.sr)

                org, rev, phase=self.audiosToPolar(orgAudio, revAudio)
                for i in range(org.shape[1]):
                    tempOrg=org[:, i*256:(i+1)*256]
                    if tempOrg.shape[1]==256:
                        saveSpectrogram(tempOrg, name=self.specDir+"org_"+idx+"_{}".format(i))
                        saveSpectrogram(rev[:, i*256:(i+1)*256], name=self.specDir+"rev_"+idx+"_{}".format(i))
                        np.save(self.specDir+idx+"_{}".format(i)+".npy", phase[:, i*256:(i+1)*256])



                with open(self.specDir+"progress", 'w') as f:
                    f.write(idx)

                pbar.update()

