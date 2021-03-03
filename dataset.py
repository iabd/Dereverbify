from glob import glob
from os import listdir
from os.path import splitext
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class BasicDataset(Dataset):
    def __init__(self, specDir, sr=16000):
        self.dir =specDir
        self.sr =sr
        self.ids =[splitext(file[4:])[0] for file in listdir(self.dir) if file.startswith('org_')]

    def __len__(self):
        return len(self.ids)

    def toImage(self, tensor):
        plt.imshow(tensor.numpy().astype(np.int16).transpose((1, 2, 0)))

    @classmethod
    def preprocess(cls, img, size=(256, 256)):
        img =img.resize(size)
        return np.array(img)


    def __getitem__(self, i):
        idx =self.ids[i]
        original =glob(self.dir +"org_" +idx +".npy")[0]
        reverbed =glob(self.dir +"rev_" +idx +".npy")[0]
        transform =transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        original=np.load(original)
        reverbed =np.load(reverbed)

        return {
            'original': transform(original),
            'reverbed': transform(reverbed)
        }


# if __name__=="__main__":
#     path="/Users/zombie/Downloads/LJSpeech-1.1/specs/"
#     t=BasicDataset(path)
#     print(t[0])
    # breakpoint()