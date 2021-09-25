import os, random, argparse, librosa
from tqdm.contrib.concurrent import process_map
from utils import clipRevAudio
import pyroomacoustics as pra
import soundfile as sf


def reverbify(audio, targetFile, roomDim, rt60, sr):
    """
    This function uses Sabine's formula to estimate the reverberation time.

    :param audio: audio file
    :param rt60:  1s is optimum for a lecture hall, 2-2.25 for concert hall.
    :param roomDim: lxbxh dimension of room
    :return: reverbed audio
    """
    orgAudio, _ = librosa.load(audio, sr)

    energyAbsorption, maxOrder = pra.inverse_sabine(rt60, roomDim)
    m = pra.Material(energy_absorption=energyAbsorption)
    room = pra.ShoeBox(roomDim, fs=16000, materials=m, max_order=maxOrder)
    room.add_source([2.5, 3.73, 1.76], signal=orgAudio, delay=0.4)

    micLocation = np.c_[
        [6.3, 4.87, 1.2],
    ]
    room.add_microphone_array(micLocation)

    room.compute_rir()
    room.simulate()

    revAudio = room.mic_array.signals[0]
    revAudio = clipRevAudio(orgAudio, revAudio)
    sf.write(targetFile, revAudio, sr)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="/Users/zombie/Downloads/LJSpeech-1.1/wavs/")
    parser.add_argument('--target', type=str, default="/Users/zombie/Downloads/LJSpeech-1.1/reverbedWavs/")
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--format', type=str, default=".wav")
    parser.add_argument('--workers', type=int, default=os.cpu_count()-1)

    args = parser.parse_args()
    datafiles=[]
    targetfiles=[]
    rt60Values=[]
    samplingRate=[]
    roomDims=[]
    for i in os.listdir(args.data):
        if i.endswith(args.format):
            datafiles.append(args.data+i)
            targetfiles.append(args.target+i)
            roomDims.append([10, 10, 8])
            rt60Values.append(random.uniform(0.3, 1.8))
            samplingRate.append(args.sr)

    r=process_map(reverbify, *(datafiles, targetfiles, roomDims, rt60Values, samplingRate), max_workers=args.workers, chunksize=2)

    print("DONE!!!")



