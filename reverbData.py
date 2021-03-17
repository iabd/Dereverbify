import os, random, argparse, librosa
from utils import reverbify
from tqdm.contrib.concurrent import process_map

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

    r=process_map(reverbify, *(datafiles, targetfiles, roomDims, rt60Values, samplingRate), max_workers=args.workers)

    print("DONE!!!")



