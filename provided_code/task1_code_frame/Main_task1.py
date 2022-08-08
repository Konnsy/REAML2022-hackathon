from raw_dataset import RawDataset
import copy
from helper import eval_model
import numpy as np
import os
import random
import statistics
import torch
import time
from data_paths import get_pos_train_sets, get_neg_train_sets, get_pos_val_sets, get_neg_val_sets
from classifier2d import Classifier
from preproc_transform import PreprocessingTransform

# What you can do to improve results:
# • have a look at the classifier, of course
# • modify meta parameters: adapt optimizer, criterion, learning rate, number of epochs ...
# • augment training data
# • change other things that seem promising (except for the RawDataset and the window size)

windowSize = 60
lr=1E-4               # learning rate 0.0001 #1E-4, 5E-5
numEpochs = 10         # number of epoch run in total

# determine and use the CUDA-compatible graphics if one is available
# otherwise the CPU will be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize positive and negative training targets
target_pos = torch.ones((1,1)).to(device)
target_neg = torch.zeros((1,1)).to(device)

# initialize values to determine and write the best model
bestModel = None
bestLoss = 1E10
bestEpoch = 0
outFolder = str(time.time())

# load and transform the data sets
trf = PreprocessingTransform(device)
datasets_train_pos = [RawDataset(dsp, trf, windowSize) for dsp in get_pos_train_sets()]
datasets_train_neg = [RawDataset(dsp, trf, windowSize) for dsp in get_neg_train_sets()]
datasets_val_pos = [RawDataset(dsp, trf, windowSize) for dsp in get_pos_val_sets()]
datasets_val_neg = [RawDataset(dsp, trf, windowSize) for dsp in get_neg_val_sets()]

# prepare the training system
model = Classifier().to(device)
model.train()
params = [p for p in model.parameters() if p.requires_grad]
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(params, lr=lr)

# initialize a list of dataset indices for the training
idcs_pos_ds = list(range(len(datasets_train_pos)))
idcs_neg_ds = list(range(len(datasets_train_neg)))

# actually train the model
for epoch in range(numEpochs):
    losses = []
    model.train()
    
    # shuffle the dataset indices to start with random datasets
    random.shuffle(idcs_pos_ds)
    random.shuffle(idcs_neg_ds)

    # take random datasets and start with a random index in the dataset
    # => randomization + speed-up through caching effects
    for idx_ds in range(max(len(idcs_pos_ds), len(idcs_neg_ds))):
        ds_pos_select = datasets_train_pos[idcs_pos_ds[idx_ds%len(idcs_pos_ds)]]
        ds_neg_select = datasets_train_neg[idcs_neg_ds[idx_ds%len(idcs_neg_ds)]]

        start_idx = random.randint(0, max(len(ds_pos_select), len(ds_neg_select)))
        for idx_inner in range(max(len(ds_pos_select), len(ds_neg_select))):
            idx_inner += start_idx

            # load the actual single image to the device
            img_pos = ds_pos_select[idx_inner%len(ds_pos_select)].to(device)
            img_neg = ds_neg_select[idx_inner%len(ds_neg_select)].to(device)

            # make a prediction for both, a positive and a negative example, and optimize the model parameters
            pred_pos = model(img_pos)
            del img_pos
            loss_pos = criterion(pred_pos, target_pos)
            loss_pos.backward()
            optimizer.step()
            optimizer.zero_grad()
            del pred_pos
        
            pred_neg = model(img_neg)
            del img_neg
            loss_neg = criterion(pred_neg, target_neg)
            loss_neg.backward()
            optimizer.step()
            optimizer.zero_grad()
            del pred_neg

    # evaluate the models
    model.eval()    
    losses = []

    with torch.no_grad():
        for ds in datasets_val_pos:
            for img_pos in ds:
                pred_pos = model(img_pos.to(device))
                loss_pos = criterion(pred_pos, target_pos).item()
                losses.append(loss_pos)

        for ds in datasets_val_neg:
            for img_neg in ds:
                pred_neg = model(img_neg.to(device))
                loss_neg = criterion(pred_neg, target_neg).item()
                losses.append(loss_neg)

        avgLoss = statistics.mean(losses)

        # check if the current model improved results
        # and save a copy it if is
        if bestModel is None or bestLoss > avgLoss:            
            bestModel = copy.deepcopy(model).cpu()
            bestLoss = avgLoss
            bestEpoch = epoch

            # write the current optimimum to disk (with a time stamp in order to keep multiple runs separated)
            os.makedirs(outFolder, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(outFolder, "model_{:d}_valLoss{:.3f}.pt".format(epoch, avgLoss)))
            
    print("epoch {:d}: {:.5f}".format(epoch, avgLoss))

    # calculate and print the accuracy (= share-of-correct-pos + share-of-correct-neg)/2
    acc, [tp, tn, fp, fn] = eval_model(model, None, device, datasets_val_pos, datasets_val_neg)
    print("acc: {:.3f}% | tp: {:d}, tn: {:d}, fp: {:d}, fn: {:d}".format(
        acc*100, tp, tn, fp, fn))