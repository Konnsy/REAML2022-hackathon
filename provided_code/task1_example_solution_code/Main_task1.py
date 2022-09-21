from raw_dataset import RawDataset
import copy
from helper import eval_model
import numpy as np
import os
import os.path as osp
import random
import statistics
import torch
import time
from data_paths import get_pos_train_sets, get_neg_train_sets, get_pos_val_sets, get_neg_val_sets, get_test_sets
from classifier2d import Classifier
from preproc_transform import PreprocessingTransform
from torchvision import transforms

windowSize = 60
lr=5E-5                 # learning rate
numEpochs = 10          # number of epoch run in total

# restrict the max. number of examples to train/evaluate per dataset in one epoch
maxPerDataset = 50 # lower values to speed up, higher to be more accurate

# determine and use the CUDA-compatible graphics if one is available
# otherwise the CPU will be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load and transform the data sets
trf = PreprocessingTransform(device)

# prepare the training system
model = Classifier().to(device)

if not osp.exists("model.pt"):
    # initialize positive and negative training targets
    target_pos = torch.ones((1,1)).to(device)
    target_neg = torch.zeros((1,1)).to(device)

    # initialize values to determine and write the best model
    bestModel = None
    bestAcc = 0
    bestEpoch = 0

    outFolder = str(time.time())
    trf_train = trf
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(params, lr=lr)

    datasets_train_pos = [RawDataset(dsp, trf_train, windowSize) for dsp in get_pos_train_sets()]
    datasets_train_neg = [RawDataset(dsp, trf_train, windowSize) for dsp in get_neg_train_sets()]
    datasets_val_pos = [RawDataset(dsp, trf, windowSize) for dsp in get_pos_val_sets()]
    datasets_val_neg = [RawDataset(dsp, trf, windowSize) for dsp in get_neg_val_sets()]

    # initialize a list of dataset indices for the training
    idcs_pos_ds = list(range(len(datasets_train_pos)))
    idcs_neg_ds = list(range(len(datasets_train_neg)))

    # actually train the model
    for epoch in range(numEpochs):
        print(f"epoch {epoch}")
        losses = []
        model.train()
    
        # shuffle the dataset indices to start with random datasets
        random.shuffle(idcs_pos_ds)
        random.shuffle(idcs_neg_ds)

        tBegin = time.time()

        # take random datasets and start with a random index in the dataset
        # => randomization + speed-up through caching effects
        for idx_ds in range(max(len(idcs_pos_ds), len(idcs_neg_ds))):
            ds_pos_select = datasets_train_pos[idcs_pos_ds[idx_ds%len(idcs_pos_ds)]]
            ds_neg_select = datasets_train_neg[idcs_neg_ds[idx_ds%len(idcs_neg_ds)]]

            start_idx = random.randint(0, max(len(ds_pos_select), len(ds_neg_select)))            
            nMax = min(maxPerDataset, max(len(ds_pos_select), len(ds_neg_select)))
            for idx_inner in range(nMax):
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

                losses.append(loss_pos.item())
                losses.append(loss_neg.item())

        tDiffSum = (time.time() - tBegin)
        tDiff = tDiffSum * 1000.0 / len(losses)  # in ms per example
        print("training took {:.1f}ms per example ({:d}s in total)".format(tDiff, int(tDiffSum)))
        print("loss: {:.2f}".format(statistics.mean(losses)))

        # evaluate the models
        model.eval()
        tp, fn, tn, fp = 0, 0, 0, 0

        tBegin = time.time()
        acc, [tp, tn, fp, fn] = eval_model(
            model, None, device, datasets_val_pos, datasets_val_neg, maxPerSet=maxPerDataset)
                                                
        # calculate and print the accuracy (= share-of-correct-pos + share-of-correct-neg)/2
        acc = (tp/(tp+fn) + tn/(tn+fp)) * 0.5
        print("acc: {:.3f}% | tp: {:d}, tn: {:d}, fp: {:d}, fn: {:d}".format(
            acc*100, tp, tn, fp, fn))

        tTot = time.time() - tBegin
        tDiff = tTot * 1000.0 / (tp+tn+fp+fn)  # in ms per example
        print("validation took {:.1f}ms per example ({:d}s in total)".format(tDiff, int(tTot)))

        # check if the current model improved results
        # and save a copy it if is
        if bestModel is None or bestAcc < acc:            
            bestModel = copy.deepcopy(model).cpu()
            bestAcc = acc
            bestEpoch = epoch

            # write the current optimimum to disk (with a time stamp in order to keep multiple runs separated)
            os.makedirs(outFolder, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(outFolder, "model_{:d}_acc{:.3f}.pt".format(epoch, acc)))
            
    print("best acc: {:.3f} in epoch {:d}".format(bestAcc, bestEpoch))

else:
    # in this path the model was already prepared (no training run)
    # the following code will produce the results file for all test datasets
    model.load_state_dict(torch.load("model.pt"))

    if len(get_test_sets()) > 0:  
        with torch.no_grad():
            test_sets = [RawDataset(dsp, trf, windowSize) for dsp in get_test_sets()]

            with open("test_results.txt", "w") as file:
                set_names = [osp.basename(p) for p in get_test_sets()]
                for set_name, ds in zip(set_names, test_sets):
                    print(set_name)
                    count_pos = 0
                    count_neg = 0
                    for img in ds:
                        img = img.to(device)
                        pred = model(img)
                        pred = pred.item()
                        if pred > 0.5:
                            count_pos += 1
                        else:
                            count_neg += 1
                
                    file.write(f"{count_pos>count_neg}, {count_pos}, {count_neg}, {set_name}\n")
                    file.flush()