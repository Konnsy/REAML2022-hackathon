"""
required libraries: 
    * numpy (pip install numpy)
    * pillow (pip install Pillow)
    * torch (https://pytorch.org/get-started/locally/)    
"""

import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from raw_dataset import RawDataset
from cvat_loader_util import CVATLoaderUtil

class ParticlesDataset(Dataset):

    def __init__(self, datasetFolder, windowSize=100, transform=None, target_transform=None):
        self.datasetFolder = datasetFolder
        self.windowSize = windowSize
        self.transform = transform
        self.target_transform = target_transform
        self.pathAnnotations = osp.join(self.datasetFolder, "annotations.xml")

        self.cvatLoader = CVATLoaderUtil(self.pathAnnotations)
        self.blockLoader = RawDataset(self.datasetFolder, transform, windowSize=windowSize)


    def __getitem__(self, index):

        fileName = self.blockLoader.filePathByIndex(index, onlyName=True)
        block = self.blockLoader[index]

        if fileName in self.cvatLoader.annotationData:
            mask = self.cvatLoader.annotationData[fileName][-1].unsqueeze(0).unsqueeze(0)
        else:
            mask = torch.zeros((1, 1, block.shape[-2], block.shape[-1]))

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return block, mask

    def __len__(self):
        return len(self.windowLoader)

