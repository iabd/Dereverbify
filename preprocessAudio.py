import librosa
import pyroomacoustics as pra
from matplotlib import cm
import numpy as np
import pylab, os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def fetchProgress(dir, ids):
    try:
        with open(dir+"progress", 'r') as f:
            p=f.readline()
            return ids.index(p)+1
    except:
        return 0



def clipRevAudio(org, rev):
    if np.nonzero(org)[0][0]<np.nonzero(rev)[0][0]:
        return rev[np.nonzero(rev)[0][0]:]
    return rev

def saveSpectrogram(s, name):
    pylab.figure(figsize=(3,3))
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    librosa.display.specshow(s, y_axis='hz', x_axis='time', cmap=cm.jet)
    pylab.savefig('{}.jpg'.format(name), bbox_inches=None, pad_inches=0)
    pylab.close()

def reverbify(audio, rt60=0.6, roomDim=[10, 10, 8]):

    """
    This function uses Sabine's formula to estimate the reverberation time.

    :param audio: audio file
    :param rt60:  1s is optimum for a lecture hall, 2-2.25 for concert hall.
    :param roomDim: lxbxh dimension of room
    :return: reverbed audio
    """
    orgAudio, _= librosa.load(audio, 22050)


    energyAbsorption, maxOrder=pra.inverse_sabine(rt60, roomDim)
    m = pra.Material(energy_absorption=energyAbsorption)
    room = pra.ShoeBox(roomDim, fs=16000, materials=m, max_order=maxOrder)
    room.add_source([2.5, 3.73, 1.76],signal=orgAudio, delay=0.4)

    micLocation=np.c_[
        [6.3, 4.87, 1.2],
    ]
    room.add_microphone_array(micLocation)

    room.compute_rir()
    room.simulate()

    revAudio=room.mic_array.signals[0]
    revAudio=clipRevAudio(orgAudio, revAudio)

    return revAudio





# reverbify('/Users/zombie/Downloads/LJSpeech-1.1/wavs/LJ001-0002.wav')


