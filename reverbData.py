import scipy, os, random, sys, argparse
from tqdm import tqdm
import numpy as np
from preprocessAudio import reverbify
from scipy.io import wavfile






if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="/Users/zombie/Downloads/LJSpeech-1.1/wavs/")
    parser.add_argument('--target', type=str, default="/Users/zombie/Downloads/LJSpeech-1.1/reverbedWavs/")
    parser.add_argument('--sr', type=int, default=22050)


    args = parser.parse_args()
    files=os.listdir(args.data)

    with tqdm(total=len(files), position=0, leave=True) as pbar:
        for idx, file in enumerate(tqdm(files, position=0, leave=True)):
            if file.endswith(".wav"):
                print("{}/{}".format(idx+1, len(files)), end="\r")

                revAudio=reverbify(args.data+file, rt60=random.uniform(0.3,0.8))
                scipy.io.wavfile.write(args.target+file, args.sr, revAudio.astype(np.int16))
                pbar.update()

    print("DONE!!!")



