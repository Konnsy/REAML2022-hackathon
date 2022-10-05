import numpy as np
import cv2
import torch

class Detector:
    def __init__(self, minArea=16):       
        super().__init__()

        # Setup SimpleBlobDetector parameters.
        self.params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        self.params.minThreshold = 0
        self.params.maxThreshold = 127

        self.params.filterByArea = True
        self.params.minArea = minArea

        self.params.filterByCircularity = False
        self.params.filterByConvexity = False
        self.params.filterByInertia = False

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            self.detector = cv2.SimpleBlobDetector(self.params)
        else : 
            self.detector = cv2.SimpleBlobDetector_create(self.params)
        
    def detect(self, img):
        img = (img*255)
        img = img.view(img.shape[-2:]).detach().cpu().numpy().astype(np.uint8)

        keypoints = self.detector.detect(~img)

        boxes = []
        for k in keypoints:
            box = [ int(k.pt[0]-k.size), int(k.pt[1]-k.size), 
                    int(k.pt[0]+k.size), int(k.pt[1]+k.size)]
            boxes.append(box)
                
        return boxes


