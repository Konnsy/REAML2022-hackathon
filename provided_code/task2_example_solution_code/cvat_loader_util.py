"""
required libraries:
    * matplotlib (pip install matplotlib)
    * numpy (pip install numpy)
    * torch (https://pytorch.org/get-started/locally/)
    * xmltodict (pip install xmltodict)
    * PIL (pip install Pillow)
"""

import numpy as np
import os.path as osp
import xmltodict
import collections
import torch
import sys
import copy
import math
import statistics
from matplotlib.path import Path
from PIL import Image, ImageDraw 
from helper import drawEllipseFromBox
from helper import maskImgFromPolygons
from helper import boxIoU
from helper import calcCliques
from helper import mergeableBoxCliquesByIoU
from tracing import BoxTracer

class CVATLoaderUtil:
    """
    Important functions:
        • getFileNames: return all file names mentioned in the annotations
        • getBoxesByFileName: return boxes for a certain fileName
        • getMaskByFileName: return the pixel mask for a certain fileName
        • convertToOtherWindowSize: convert annotations (from their default size of 100) to a given window size
    """
    def __init__(self, xmlPath, boxesToEllipses=False, allToEllipses=False):

        self.xmlPath = xmlPath
        self.height = 0
        self.width = 0
        self.minId = -1
        self.maxId = -1
        self.owner = ''
        self.annotationData = collections.OrderedDict() # fileName -> [id, polygons, boxes, mask]

        if not osp.exists(xmlPath):
            print("Error: Cannot find {}".format(xmlPath))
            return
        
        self.loadAnnotations()
        self.removeDuplicates()

        if boxesToEllipses or allToEllipses:
            self.boxesToEllipses(alsoPolys=allToEllipses)

    def convertToOtherWindowSize(self, newWindowSize):
        if newWindowSize == 100:
            return

        self.stretch = newWindowSize / 100
        self.convertAnnotationLengthsByFactor()

    def boxesToEllipses(self, alsoPolys=False):
        for fileName in self.annotationData.keys():
            [id, polygons, boxes, mask] = self.annotationData[fileName]
            if len(polygons)==0 and len(boxes)>0:
                polygons = [ [] for _ in boxes]

            mergedMask = np.zeros_like(mask)
            modPolygons = []

            for poly, box in zip(polygons, boxes):
                box = [box[1], box[0], box[3], box[2]]
                if len(poly)==0 or (len(poly) in list(range(3,8))) or alsoPolys:
                    # polygon representing a box
                    mergedMask, polyEl = drawEllipseFromBox(mergedMask, box)
                    modPolygons.append(polyEl)
                else:
                    # not representing a box
                    mask = maskImgFromPolygons([poly], mergedMask.shape[0], mergedMask.shape[1])
                    mergedMask = ((mask + mergedMask) > 0).astype(np.uint8)
                    modPolygons.append(poly)
    
            self.annotationData[fileName] = [id, modPolygons, boxes, mergedMask]

    def getAnnotationArea(self):
        """
        Get the average area one annotation polygon covers
        """
        areas = []

        for fileName in self.annotationData.keys():
            maskShape = self.annotationData[fileName][3].shape

            for poly in self.annotationData[fileName][1]:
                mask = (maskImgFromPolygons([poly], maskShape[0], maskShape[1]) > 0).astype(np.uint8)
                areas.append(mask.sum())

        ares = list(filter(lambda a : a>0, areas))

        medArea = statistics.median(areas)
        minArea = min(areas)
        maxArea = max(areas)
        return medArea, minArea, maxArea

    def convertAnnotationLengthsByFactor(self):        
        iouThreshold = 0.1
        distTolerance = 5
        minLenTraces = 5
         
        if self.stretch <= 0:
            raise ValueError("window size factors must be values > 0!")

        factor = self.stretch
        fileNames = list(self.annotationData.keys())

        # calculate traces of the current annotations
        bbt = self.boxTracerFromAnnotations(fileNames, minLenTraces, iouThreshold, distTolerance)
        oldTraces = bbt.calculateTraces(fillGaps=True)
              
        annot = copy.deepcopy(self.annotationData)

        # fileName -> [id, polygons, boxes, mask]
        self.annotationData = collections.OrderedDict()
        for fileName in fileNames:
            self.annotationData[fileName] = [id, [], [], np.zeros((self.width, self.height), dtype=np.uint8)]

        for framesOld, boxesOld in oldTraces:          
            assert len(boxesOld) > 0

            # old meta info for the whole trace
            startOld = min(framesOld)
            endOld = max(framesOld)
            spanOld = endOld - startOld     
            centerOld = int((startOld + endOld) * 0.5)
            
            # new meta info for the whole trace
            centerNew = centerOld
            spanNew = int(spanOld * factor + 0.5)
            startNew = centerNew - math.ceil(spanNew * 0.5)
            endNew = centerNew + math.floor(spanNew * 0.5)

            centerNew = max(0, min(len(fileNames)-1, centerNew))
            startNew = max(0, min(len(fileNames)-1, startNew))
            endNew = max(0, min(len(fileNames)-1, endNew))

            for frameNew in range(startNew, endNew):
                relPosition = float(frameNew - startNew) / spanNew
                newName = fileNames[frameNew]                
                frameOld = int(startOld * (1.0 - relPosition) + endOld * relPosition)
                frameOldInner = frameOld - startOld
                oldName = fileNames[frameOld]

                # smoothen annotations
                colFrom = max(0, frameOldInner - 2)
                colTo = min(spanOld, frameOldInner + 2)
                #print(f"{colFrom}-{colTo}")
                colBoxes = [boxesOld[frcb] for frcb in range(colFrom, colTo+1)]
                colBox = [
                            statistics.mean([c[0] for c in colBoxes]),
                            statistics.mean([c[1] for c in colBoxes]),
                            statistics.mean([c[2] for c in colBoxes]),
                            statistics.mean([c[3] for c in colBoxes]),
                         ]

                # elements: id, polygons, boxes, mask
                self.annotationData[newName][2].append(colBox)

        self.boxesToEllipses(alsoPolys=True)
        
        fileNames = list(self.annotationData.keys())
        for fileName in fileNames:
            polys = self.annotationData[fileName][1]
            boxes = self.annotationData[fileName][2]
            boxesNew = []
            polysNew = []

            # take the first box and poly of each clique of boxes
            # determined by their iou values
            cliques = mergeableBoxCliquesByIoU(boxes, iouThreshold=iouThreshold)            
            for clique in cliques:
                boxesNew.append(boxes[clique[0]])
                polysNew.append(polys[clique[0]])                

            self.annotationData[fileName][1] = polysNew
            self.annotationData[fileName][2] = boxesNew

    def getPolygonsPerFrame(self):
        polygonsPerFrame = []

        for fileName in self.annotationData.keys():
            polygonsPerFrame.append(self.annotationData[fileName][1])

        return polygonsPerFrame

    def getFileNames(self):
        return list(self.annotationData.keys())

    def getFirstFileName(self):
        names = sorted(self.annotationData.keys())
        if len(names) > 0:
            return names[0]

    def getLastFileName(self):
        names = sorted(self.annotationData.keys())
        if len(names) > 0:
            return names[-1]
        
    def hasData(self):
        return len(self.annotationData) > 0
        
    def hasEntryForFileName(self, fileName):
        return fileName in self.annotationData
    
    def getPolygonsByImgName(self, fileName):
        try:
            return self.annotationData[fileName][1]
        except:
            return []

    def getBoxesByImgName(self, fileName):
        return self.annotationData[fileName][2]

    def getShape(self):
        return (self.width, self.height)
    
    def getBoxesPerFrame(self):
        boxesPerFrame = []
        for fileName in self.getFileNames():
            boxesPerFrame.append(self.annotationData[fileName][2])
        return boxesPerFrame

    def getBoxesByFileName(self, fileName):
        if self.hasEntryForFileName(fileName):
            return self.annotationData[fileName][2]
        else:
            return []
    
    def getMaskByFileName(self, fileName):
        if (self.hasEntryForFileName(fileName) 
            and len(self.annotationData[fileName]) > 0
            and len(self.annotationData[fileName][3])>0):
            mask = self.annotationData[fileName][3]
            mask = mask.transpose(1, 0)
            return mask
        else:
            mask = np.zeros((self.width, self.height), dtype=np.uint8)
            return mask
        
    def getSeparateMasksByFileName(self, fileName):
        masks = []

        for poly in self.annotationData[fileName][1]:
            mask = maskImgFromPolygons([poly], self.width, self.height)
            masks.append(mask)

        return masks

    def getMaskByIdx(self, idx):
        keys = list(self.annotationData.keys())
        if len(keys) > idx:
            fileName = keys[idx]
            if self.hasEntryForFileName(fileName):
                return self.annotationData[fileName][3]

        return np.zeros((self.width, self.height), dtype=np.uint8)

    def boxTracerFromAnnotations(self, fileNames, minLenTraces, iouThreshold, distTolerance):
        fileNames = sorted(fileNames)

        bbt = BoxTracer(minLenTraces, iouThreshold, distTolerance)
        for fileName in fileNames:
            if fileName in self.annotationData:
                bbt.addToTraces(self.getBoxesByFileName(fileName))
            else:
                # no annotation in this frame
                bbt.addToTraces([])
                
        return bbt


    def loadAnnotations(self):
        mergedMask = None

        try:
            with open(self.xmlPath) as fd:
                doc = xmltodict.parse(fd.read())

            imgInfo = []

            self.owner = doc['annotations']['meta']['task']['owner']['username']

            for imgEntry in doc['annotations']['image']:
                self.height = int(imgEntry['@height'])
                self.width = int(imgEntry['@width'])
                fileName = imgEntry['@name']
                id = int(imgEntry['@id'])
                
                if self.minId == -1:
                    self.minId = id
                    self.maxId = id
                else:
                    self.minId = min(id, self.minId)
                    self.maxId = max(id, self.maxId)
            
                polygons = []
                if 'polygon' in imgEntry:
                    polyEntries = imgEntry['polygon']

                    if '@label' in polyEntries:
                        pointsString = polyEntries['@points']
                        points = []

                        for pointString in pointsString.split(';'):
                            # store this point as a list of float values
                            points.append(list(map(lambda e : float(e), pointString.split(','))))

                        polygons.append(points)
                    else:
                        for polyEntry in polyEntries:
                            pointsString = polyEntry['@points']
                            points = []

                            for pointString in pointsString.split(';'):
                                points.append([int(float(e)) for e in pointString.split(',')])

                            polygons.append(points)

                boxes = []

                if len(polygons) > 0:
                    for poly in polygons:
                        xPos = [points[0] for points in poly]
                        xPos.sort()
                        yPos = [points[1] for points in poly]
                        yPos.sort()
                        xMin, xMax = xPos[0], xPos[-1]
                        yMin, yMax = yPos[0], yPos[-1]
                        boxes.append([xMin, yMin, xMax, yMax])

                mask = maskImgFromPolygons(polygons, self.width, self.height)

                assert len(polygons) == len(boxes)
                self.annotationData[fileName] = [id, polygons, boxes, mask]

        except:
           print("Error: Exception during loading of {}".format(self.xmlPath))
           return

        if len(self.annotationData) == 0:
            print("Warning: There was no exception, but no fitting annotations were found in {}".format(self.xmlPath))
                                
            
    def removeDuplicates(self, iouThreshold=0.5):

        for fileName in self.annotationData.keys():
            [id, polygons, boxes, mask] = self.annotationData[fileName]

            # collect intersecting box pairs
            pairs = []
            for i in range(len(boxes)):
                for j in range(i+1, len(boxes)):
                    iou = boxIoU(boxes[i], boxes[j])
                    if iou >= iouThreshold:
                        pairs.append([i,j])

            cliques = calcCliques(pairs, list(range(len(boxes))))

            toRemove = []
            for clique in cliques:
                if len(clique) > 1:
                    toRemove.extend(clique[1:])

            toRemove = list(set(toRemove))
            toRemove.sort(reverse=True)
            
            for idxToRem in toRemove:
                del polygons[idxToRem]
                del boxes[idxToRem]

    def __len__(self):
        return len(list(self.annotationData.keys()))

