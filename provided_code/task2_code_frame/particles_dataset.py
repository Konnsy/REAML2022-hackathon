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
from tracing import BoxTracer
import cv2

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

        self.blockLoader = RawDataset(self.datasetFolder, transform, windowSize=windowSize)
        if len(self.blockLoader) == 0:
            raise ValueError(f"No files found at {self.datasetFolder}!")

        # load annotation data
        self.cvatLoader = CVATLoaderUtil(self.pathAnnotations)
        self.fileNames = self.cvatLoader.getFileNames()
        self.tracer = BoxTracer(minLenTraces=10, iouThreshold=0.1, distTolerance=3)
        for boxes in self.cvatLoader.getBoxesPerFrame():
            self.tracer.addToTraces(boxes)
        self.traceCount = len(self.tracer.calculateTraces())

        # convert window sizes and prepare annotations
        if windowSize != 100:
            self.cvatLoader.convertToOtherWindowSize(windowSize)
        else:
            self.cvatLoader.boxesToEllipses(alsoPolys=True)

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
        
        if index >= len(self):
            raise IndexError()

        fileName = self.blockLoader.filePathByIndex(index, onlyName=True)
        block = self.blockLoader[index]
              
        if self.cvatLoader.hasEntryForFileName(fileName):
            mask = self.cvatLoader.getMaskByFileName(fileName)
            mask = torch.from_numpy(mask.astype(np.float32))
            mask = mask.view((1, 1, *mask.shape[-2:]))
            polys = self.cvatLoader.getPolygonsByImgName(fileName)
            boxes = self.cvatLoader.getBoxesByImgName(fileName)
        else:
            mask = torch.zeros((1, 1, block.shape[-2], block.shape[-1]))
            polys = []
            boxes = torch.tensor([])
        
        mask = (mask > 0.5).float()
        res = [block, mask]

        if self.retBoxes:
            res.append(boxes)        
        if self.retPolys:
            res.append(polys)

        return res

    def filePathByIndex(self, idx):
        return self.blockLoader.filePathByIndex(idx)

    def getNumberOfTraces(self):
        return self.traceCount

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
