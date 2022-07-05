"""
required libraries: 
    * numpy (pip install numpy)
    * open cv (pip install opencv-python)
    * pillow (pip install Pillow)
    * torch (https://pytorch.org/get-started/locally/)    
"""

import cv2
import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class RawDataset(Dataset):
    """
        datasetFolder (string): directory which contains the "raw" folder
        windowSize (int): determines the number of images that are stacked into a block (=window),
                            must be higher than 0
        transform: Transform operation that is executed on the block of frames before it is returned.
                        with the default of None, the loaded block is not modified.

        return: tensor of shape (1, windowSize, x, y) where x and y are the dimensions of 
                a single image

        Example usage: 
            1. using an iterator: 
            for window in RawDataset(<datasetFolder>):
                ...

            2. using the index:
            ds = RawDataset(<datasetFolder>)
            for idx in range(len(ds)):
                window = ds[idx]
    """

    def __init__(self, datasetFolder, transform=None, windowSize=60):
        if windowSize < 1 or not isinstance(windowSize, int):
            raise ValueError("WindowSize has to be an integer higher than zero!")

        self.windowSize = windowSize
        self.transform = transform
        self.iterIdx = 0

        self.root = osp.join(datasetFolder, "raw")        
        self.filePaths = [osp.join(self.root, fn) for fn in os.listdir(self.root)]
        self.filePaths = list(filter(lambda fp : hasImageExtension(fp), self.filePaths))

        if len(self.filePaths) == 0:
            raise FileNotFoundError(f"Could not find image files in {self.root}!")
        
        # caching
        self.imgs = []
        self.startIdx = None

        print(f"loaded {datasetFolder}")


    def __getitem__(self, idx):
        """
            Args:
                index (int): Index
            Returns:
                image
        """       
        if idx >= len(self):
            raise IndexError

        # caching
        if (self.startIdx is not None) and (self.startIdx + self.windowSize > idx):
            # delete too early examples
            for _ in range(idx-self.startIdx):
                del self.imgs[0]

            # add the missing examples
            for k in range(self.startIdx+self.windowSize, idx+self.windowSize):
                self.imgs.append(
                    torch.from_numpy(cv2.imread(self.filePaths[k], 0)))

        else:
            # nothing can be re-used
            del self.imgs
            self.imgs = [torch.from_numpy(cv2.imread(fp, 0)) for fp in self.filePaths[idx:idx+self.windowSize]]
        
        self.startIdx = idx
        img = torch.stack(self.imgs, 0).float().unsqueeze(0)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def filePathByIndex(self, idx, onlyName=False):
        """
        Returns the name of the center frame of the block with the given idx
        """
        filePath = self.filePaths[idx+self.windowSize//2]

        if onlyName:
            return osp.basename(filePath)
        else:
            return filePath

    def __len__(self):
        return len(self.filePaths) - self.windowSize + 1

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.iterIdx += 1
            return self[self.iterIdx-1]
        except:
            self.iterIdx = 0
            raise StopIteration()


def hasImageExtension(fileName):
    for ext in [".png", ".tif", ".tiff"]:    
        if fileName.endswith(ext):
            return True
    return False