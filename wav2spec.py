import os, argparse, librosa
from spectrogram import Spectrogram

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--org', type=str, default="/Users/zombie/Downloads/LJSpeech-1.1/wavs/")
    parser.add_argument('--rev', type=str, default="/Users/zombie/Downloads/LJSpeech-1.1/reverbedWavs/")
    parser.add_argument('--target', type=str, default="/Users/zombie/Downloads/LJSpeech-1.1/specs/")
    parser.add_argument('--sr', type=int, default=16000)

    args = parser.parse_args()
    S=Spectrogram(args.org, args.rev, args.target)
    S.go()
