import os
import os.path as osp
import numpy as np
import torch
import cv2

def eval_model(model, outFolder, device, pos_sets, neg_sets, maxPerSet):
    if not outFolder is None:
        outFolderTP = osp.join(outFolder, 'TP')
        outFolderFP = osp.join(outFolder, 'FP')
        outFolderTN = osp.join(outFolder, 'TN')
        outFolderFN = osp.join(outFolder, 'FN')
        os.makedirs(outFolderTP, exist_ok=True)
        os.makedirs(outFolderFP, exist_ok=True)
        os.makedirs(outFolderTN, exist_ok=True)
        os.makedirs(outFolderFN, exist_ok=True)

    with torch.no_grad():
        model.eval()
        model = model.to(device)
        tp, fp, tn, fn = 0, 0, 0, 0

        for ds_val_pos in pos_sets:
            for idx, img_pos in enumerate(ds_val_pos):
                if idx == maxPerSet:
                    break
                pred_pos = model(img_pos.to(device)).item()
                pred_pos = pred_pos > 0.5
                if pred_pos:
                    tp += 1
                    if not outFolder is None:
                        cv2.imwrite(osp.join(outFolderTP, "tp_{:d}.png".format(tp)),
                                    (img_pos*255).cpu().squeeze(0).numpy())
                else:
                    fn += 1
                    if not outFolder is None:
                        cv2.imwrite(osp.join(outFolderFN, "fn_{:d}.png".format(fn)),
                                    (img_pos.cpu()*255).squeeze(0).numpy())

        for ds_val_neg in neg_sets:
            for idx, img_neg in enumerate(ds_val_neg):
                if idx == maxPerSet:
                    break
                pred_neg = model(img_neg.to(device)).item()
                pred_neg = pred_neg > 0.5
                if pred_neg:
                    fp += 1
                    if not outFolder is None:
                        cv2.imwrite(osp.join(outFolderFP, "fp_{:d}.png".format(fp)),
                                    (img_neg.cpu()*255).squeeze(0).numpy())
                else:
                    tn += 1
                    if not outFolder is None:
                        cv2.imwrite(osp.join(outFolderTN, "tn_{:d}.png".format(tn)),
                                    (img_neg.cpu()*255).squeeze(0).numpy())

        acc = (tp/(tp+fn) + tn/(tn+fp)) * 0.5
        return acc, [tp, tn, fp, fn]


def getAllWithRawFolders(superfolder, recursive=True):
    """
    determine the paths of datasets by searching for 'raw' subfolders
    """
    if not osp.exists(superfolder):
        return []

    withRawfolders = list(filter(lambda fd : osp.isdir(osp.join(superfolder, fd)) and
                                 'raw' in list(os.listdir(osp.join(superfolder, fd))), os.listdir(superfolder)))
    withRawfolders = list(map(lambda fd : osp.join(superfolder, fd), withRawfolders))

    if recursive:
        for fp in os.listdir(superfolder):
            fn = osp.join(superfolder, fp)
            if osp.isdir(fn):
                withRawfolders.extend(getAllWithRawFolders(fn, recursive=True))

    return withRawfolders