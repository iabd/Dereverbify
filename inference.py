import librosa, torch, IPython, io, PIL, librosa, json
import numpy as np
from utils import saveSpectrogram, complexToPolar, reconstructAudioFromBatches
import soundfile as sf


def transform(X):
    return (X - X.min()) / (X.max() - X.min())


def split(array):
    """Accepts numpy array of shape [x, y] returns squared arrays split along y-axis"""
    xdim = array.shape[0] // 2 * 2
    ydim = array.shape[1]
    ydimHead = ydim // xdim * xdim  # y-dim of the initial batches
    ydimTail = ydim - ydim // xdim * xdim  # y-dim of the last batch
    batches = np.asarray(np.hsplit(array[:xdim, :ydimHead], ydim // xdim))

    # pad the last batch with minimum value of spectrogram
    tailBatch = array[:xdim, ydimHead:]
    tailB = np.full((xdim, xdim), array.min())
    tailB[:, :ydimTail] = tailBatch
    batches = np.concatenate((batches, np.expand_dims(tailB, axis=0)), axis=0)
    batches = transform(batches)
    return np.expand_dims(batches, axis=1)



def inference(net, audio, stftParams, genAudioFile="generated.wav", checkpoint=None):
    if checkpoint:
        from unet import UNet
        net=UNet(1,1)
        ckp = torch.load(checkpoint, map_location='cpu')
        net.load_state_dict(ckp['modelStateDict'])
        net.eval()
        ckp=0

    istftParams={key:val for key, val in stftParams.items() if key!="n_fft"}
    stft=librosa.stft(audio, **stftParams)
    mag, phase=complexToPolar(stft)
    magBatch=split(mag)
    phaseBatch=split(phase)
    newMag=net(torch.from_numpy(magBatch))
    spec=saveSpectrogram(magBatch[0][0], newMag[0][0].detach(), name="generatedImage.jpg")
    genAudio=reconstructAudioFromBatches(newMag.detach().numpy(), phaseBatch, istftParams)
    sf.write(genAudioFile, np.asarray(genAudio)*10, 16000)
    return genAudioFile, spec




