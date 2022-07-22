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
    """
    Dataset to access images of an annotated recording.
    """
    def __init__(self, 
                 datasetFolder, 
                 windowSize=100, 
                 transform=None,
                 boxes=False,
                 polys=False
                 ):
        """
        datasetFolder (string): 
            path to the folder containing the 'raw' subfolder and the 'annotations.xml' file
        windowSize (int):
            The size of the loaded raw image blocks. Must be higher than 0.
        transform (transform object):
            Applied on the loaded block of raw images. NOT applied to the annotations.
        boxes (bool):
            Also boxes around annotated objects will be returned.
        polys (bool):
            Also polygons around annotated objects will be returned.
        """
        self.datasetFolder = datasetFolder
        self.windowSize = windowSize
        self.transform = transform
        self.pathAnnotations = osp.join(self.datasetFolder, "annotations.xml")
        self.retBoxes = boxes
        self.retPolys = polys

        self.cvatLoader = CVATLoaderUtil(self.pathAnnotations)
        self.blockLoader = RawDataset(self.datasetFolder, transform, windowSize=windowSize)
        self.iterIdx = 0

    def __getitem__(self, index):
        """
        Returns a loaded block or the transformed version of it (if the Dataset was given
        a transform in its constructor) and the corresponding annotations.

        Return formats:
        network input data:
            raw images of format (1, w, x, y) with window size w and image shape (x,y)
            or the transformed version of this format if a transform was set
        boxes: 
            tensor of the shape (<number of boxes, 4>) with coordinates (TODO) in each box
        polygons: 
            list containing a list of (x,y)-points defining all node positions of the polygon
    
        if box and polys are set to False:
            returns [ transform(raw block), pixel mask]

        if only boxes or only polys is set to True:
            returns [ transform(raw block), pixel mask, <boxes or polygons>]

        if both, boxes and polys, are set to True:
            returns [ transform(raw block), pixel mask, boxes, polygons]

        """
        fileName = self.blockLoader.filePathByIndex(index, onlyName=True)
        block = self.blockLoader[index]

        if fileName in self.cvatLoader.annotationData:
            mask = self.cvatLoader.getMaskByFileName(fileName).unsqueeze(0).unsqueeze(0)            
            polys = self.cvatLoader.getPolygonsByImgName(fileName)
            boxes = self.cvatLoader.getBoxesByImgName(fileName)
        else:
            mask = torch.zeros((1, 1, block.shape[-2], block.shape[-1]))
            polys = []
            boxes = torch.tensor([])

        res = [block, mask]

        if self.retBoxes:
            res.append(boxes)        
        if self.retPolys:
            res.append(polys)

        return res

    def filePathByIndex(self, idx):
        return self.blockLoader.filePathByIndex(idx)

    def __len__(self):
        return len(self.blockLoader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.iterIdx += 1
            return self[self.iterIdx-1]
        except:
            self.iterIdx = 0
            raise StopIteration()
