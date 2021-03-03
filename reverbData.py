import os, random, argparse, librosa
from tqdm import tqdm
from preprocessAudio import reverbify


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

                revAudio=reverbify(args.data+file, rt60=random.uniform(0.3,1.8))
                librosa.output.write_wav(args.target+file, revAudio,  args.sr)
                #scipy.io.wavfile.write(args.target+file, args.sr, revAudio.astype(np.int16))
                pbar.update()

    print("DONE!!!")



