import torch
from torchvision import transforms
import torchvision.transforms.functional as functional
import random

class AugmentationTransform(object):
    def __init__(self):
        self.range_contrast = (0.5, 1.5)
        self.range_scale = (0.5, 1.25)

    def __call__(self, img, mask):

        if random.random() < 0.5:
            img = transforms.functional.vflip(img)
            mask = transforms.functional.vflip(mask)

        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)

        if random.random() < 0.5:
            scale_factor = random.uniform(*self.range_scale)
            mode = random.choice(['nearest', 'bilinear'])            
            img = torch.nn.functional.interpolate(img, 
                                            size=None, 
                                            scale_factor=scale_factor, 
                                            mode=mode, 
                                            align_corners=None, recompute_scale_factor=None, antialias=False)

            mask = torch.nn.functional.interpolate(mask, 
                                            size=None, 
                                            scale_factor=scale_factor, 
                                            mode=mode, 
                                            align_corners=None, recompute_scale_factor=None, antialias=False)
                   
        if random.random() < 0.5:
            img = functional.adjust_contrast(img, contrast_factor=random.uniform(*self.range_contrast))
                  
        return img, mask