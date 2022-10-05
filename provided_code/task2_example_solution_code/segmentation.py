import torch
import torch.nn as nn
import statistics
import random
import copy
import cv2
import os
import os.path as osp
import numpy as np
import time
import shelve

class Segmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.timeStr = str(int(time.time()))
        c = 8
        self.channels = c

        self.pool = nn.AvgPool2d(2)
        self.conv_first = nn.Conv2d(1, c, 3, padding=1, dilation=1)
        self.numSkipConvLayers = 100
        self.skipConvs = nn.Sequential(*[SkipConv(c) for _ in range(self.numSkipConvLayers)])        
        self.conv_last = nn.Conv2d(c, 1, 1)
                  
    def forward(self, x):
        sIn = x.shape
        x = self.pool(x)
        x = self.conv_first(x)
        x = self.skipConvs(x)
        x = self.conv_last(x)
        x = torch.sigmoid(x)
        x = torch.nn.UpsamplingBilinear2d(size=sIn[-2:])(x)
        return x

    def train_model(self, ds_train, ds_val):
        device = next(self.parameters()).device

        from augmentation_transform import AugmentationTransform

        if len(ds_train) == 0:
            raise ValueError("Training sets must not be empty!")

        if len(ds_val) == 0:
            raise ValueError("Validation sets must not be empty!")

        print("Training Segmentor")        
        startLR = 5E-6
        schedThresh = 5
        endThres = 10
        schedFactor = 0.1
        lastSchedStep = 0
        print(startLR)
        print(self.channels)

        numPerEpoch = 100
        numEpochs = 100

        print("caching training data")
        count_ex = 0
        if not osp.exists('cache_train.dir'):
            with shelve.open('cache_train') as db_train:
                for ds in ds_train:
                    for img, mask in ds:
                        if mask.sum().item() > 0:
                            if img.shape != img.shape:
                                print(f"{ds.datasetFolder}")
                            db_train[str(count_ex)] = [img, mask]
                            count_ex += 1

        print("caching validation data")
        count_ex = 0
        if not osp.exists('cache_val.dir'):
            with shelve.open('cache_val') as db_val:
                for ds in ds_val:
                    for img, mask in ds:
                        if img.shape != img.shape:
                            print(f"{ds.datasetFolder}")
                        if mask.sum().item() > 0:
                            db_val[str(count_ex)] = [img, mask]
                            count_ex += 1
        print("caching done")

        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=startLR)
        criterion = dice_loss

        bestModel = None
        bestEpoch = None
        bestLoss = None

        db_train = shelve.open('cache_train')
        augm = AugmentationTransform()
        for epoch in range(numEpochs):
            self.train()
            tBeginEpoch = time.time()

            losses = []

            self.train()
            for count_ex in range(numPerEpoch):
                #print(f"{count_ex} of {numPerEpoch}")
                idx_ex = random.randint(0, len(db_train)-1)
                img, mask = db_train[str(idx_ex)]
                mask = mask.to(device)
                img = img.to(device)
                img, mask = augm(img, mask)
                
                pred = self(img).view(mask.shape)
                loss = criterion(pred, mask)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self.eval()
            losses_val = []
            db_val = shelve.open('cache_val')
            with torch.no_grad():
                for idx_img in range(len(db_val)):      
                        #print(f"{idx_img} of {len(db_val)}")
                        img, mask = db_val[str(idx_img)]
                        img = img.to(device)
                        mask = mask.to(device)
                        pred = self(img).view(mask.shape)
                        loss = criterion(pred, mask)                        
                        losses_val.append(loss.item())
      
            avgLoss = statistics.mean(losses_val)
            
            if bestLoss is None or avgLoss < bestLoss:
                bestLoss = avgLoss
                bestModel = copy.deepcopy(self).cpu().eval()
                bestEpoch = epoch
                os.makedirs(self.timeStr, exist_ok=True)
                self.save(osp.join(self.timeStr, "new_best_ep{:d}_loss{:.2f}.pt".format(epoch, avgLoss)))
           
            print( "epoch {:d}: {:.4f} (best so far: {:d}), took {:.2f}s".format(
                epoch, avgLoss, bestEpoch, int(time.time() - tBeginEpoch)) )

            if max(bestEpoch, lastSchedStep) + schedThresh <= epoch:
                for g in optimizer.param_groups:
                    g['lr'] *= schedFactor
                    print("reduced lr to {:.3e}".format(g['lr']))
                    lastSchedStep = epoch

            if bestEpoch + endThres < epoch:
                print("ending training")
                break
                  
        db_train.close()
        db_val.close()
        self = bestModel
        self.eval().to(device)
        print("best model set")
        

    def save(self, model_path):
        torch.save(copy.deepcopy(self).cpu().state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))


class SkipConv(nn.Module):
    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1, padding_mode='reflect')
        self.act = nn.ReLU(True)
        #self.drop = nn.Dropout(dropout_rate)
        self.norm = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1, padding_mode='reflect')

    def forward(self, x):
        x2 = self.conv1(x)
        x2 = self.norm(x2)
        x2 = self.act(x2)
        #x2 = self.drop(x2)        
        x2 = self.conv2(x2)
        return x + x2


def dice_loss(output, target):
    """
    Calculate the dice loss for a prediction 'output' and a mask 'target'.
    Only tested for single images (i.e. no batched inputs)
    """
    eps = 1e-6

    intersection = output * target
    intersection = torch.sum(intersection)
    union_intersection = torch.sum(output + target)
    dices = 1 - (2 * intersection) / (union_intersection + eps)
    mean_dice = torch.mean(dices)
    mean_dice = torch.max(torch.min(mean_dice , torch.ones_like(mean_dice)), torch.zeros_like(mean_dice))

    return mean_dice