import torch
import cv2
import numpy as np
import os
import os.path as osp
import statistics
import time
from raw_dataset import RawDataset
from segmentation import Segmenter
from detection import Detector
from tracing import BoxTracer
from particles_dataset import ParticlesDataset
from preproc_transform import PreprocessingTransform
from trace_filter import TraceFilter
from data_paths import get_train_sets, get_val_sets, get_test_sets
from helper import writeBoxesToImage, scale

# What you can do to improve results:
# • modify meta parameters of segmentation, box tracing etc.
# • augment training data
# • change other things that seem promising

def main():
    windowSize = 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare the modules of the pipeline
    seg = Segmenter().to(device)
    det = Detector()
    tra = BoxTracer(minLenTraces=10, iouThreshold=0.1, distTolerance=3)
    trf = PreprocessingTransform(device)
       
    # load an already trained model or, if not available, train a new one
    model_path = "segmenter.pt"
    if osp.exists(model_path):
        print("loading segmenter weights")
        seg.load(model_path)
        seg.eval()
        seg.to(device)

        print("preparing validation datasets")
        datasets_val = [ParticlesDataset(dsp, windowSize=windowSize, transform=trf) for dsp in get_val_sets()]
        datasets_val_raw = [RawDataset(dsp, windowSize=1) for dsp in get_val_sets()]
    else:
        print("preparing training datasets")
        datasets_train = [ParticlesDataset(dsp, windowSize=windowSize, transform=trf) for dsp in get_train_sets()]
        datasets_train_raw = [RawDataset(dsp, windowSize=1) for dsp in get_train_sets()]

        print("preparing validation datasets")
        datasets_val = [ParticlesDataset(dsp, windowSize=windowSize, transform=trf) for dsp in get_val_sets()]
        datasets_val_raw = [RawDataset(dsp, windowSize=1) for dsp in get_val_sets()]

        print("training the Segmenter")
        seg.train_model(datasets_train, datasets_val)
        seg.save(model_path)
               
    if len(datasets_val) > 0:
            print("starting validation")
            accuracies_val = []
            with torch.no_grad():
                for idx_dataset, (val_ds, val_ds_raw) in enumerate(zip(datasets_val, datasets_val_raw)):
                    print("validating on dataset {:d} of {:d} ({})".format(
                        idx_dataset+1, len(datasets_val), val_ds.datasetFolder))
                    tra.clear() # delete boxes of previous datasets

                    with torch.no_grad():
                        for _ in range(windowSize//2):
                            # add empty boxes for the first windowSize//2 frames
                            # since they cannot be used for detection
                            tra.addToTraces([])
                                    
                        for idx_img, (img, _) in enumerate(val_ds):
                            # detect particles on the single image
                            segmented = seg(img.to(device))
                            candidate_boxes = det.detect(segmented)
                            tra.addToTraces(candidate_boxes)

                            # comment in the following section to create visualizations
                            #from helper import writeBoxesToImage
                            #img_out = scale(img, (0,1))
                            #img_out = (img_out*255).view(img.shape[-2:]).cpu().numpy().astype(np.uint8)
                            #img_out = writeBoxesToImage(img_out, candidate_boxes)
                            #os.makedirs(f"out/val_{val_ds.datasetFolder}/boxes", exist_ok=True)
                            #cv2.imwrite("out/val_{}/boxes/seg_{}.png".format(val_ds.datasetFolder, idx_img), img_out)

                        # get traces from (not yet connected) boxes
                        traces = tra.calculateTraces()
  
                        # filter out traces that do not represent particles
                        traces_filt = TraceFilter(val_ds_raw).filter_traces(traces)
                        count_ds = len(traces_filt)
                
                    # calculate and print the count accuracy
                    expected_ds = val_ds.getNumberOfTraces()
                    acc = 1.0 - (abs(count_ds-expected_ds)) / max(count_ds, expected_ds)

                    print("{} acc.: found {:d} and expected {} particle traces".format(
                        acc, count_ds, expected_ds))
                    accuracies_val.append(acc)

            print("average validation acc.: {:.2f}".format(statistics.mean(accuracies_val)))

    print("preparing test datasets")
    datasets_test = [RawDataset(dsp, windowSize=windowSize, transform=trf) for dsp in get_test_sets()]
    datasets_test_raw = [RawDataset(dsp, windowSize=1) for dsp in get_test_sets()]

    if len(datasets_test) > 0:
        print("analyzing test sets")
        with open("test_results.txt", "w") as file:
            for idx_dataset, (test_ds, test_ds_raw) in enumerate(zip(datasets_test, datasets_test_raw)):
                print("testing dataset {:d} of {:d}".format(idx_dataset+1, len(datasets_test_raw)))
                print(test_ds.root)
                tra.clear() # delete the boxes of previous datasets

                tBegin = time.time()
                with torch.no_grad():
                    for _ in range(windowSize//2):
                        # add empty boxes for the first windowSize//2 frames
                        # since they cannot be used for detection
                        tra.addToTraces([])

                    for idx_img, img in enumerate(test_ds):
                        # detect particles on the single image
                        segmented = seg(img.to(device))                        
                        candidate_boxes = det.detect(segmented)
                        tra.addToTraces(candidate_boxes)

                        # comment in the following section to create visualizations
                        #from helper import writeBoxesToImage
                        #img_out = scale(img, (0,1))
                        #img_out = (img_out*255).view(img.shape[-2:]).cpu().numpy().astype(np.uint8)
                        #img_out = writeBoxesToImage(img_out, candidate_boxes)
                        #os.makedirs(f"out/test_{idx_dataset}/boxes", exist_ok=True)
                        #cv2.imwrite("out/test_{:d}/boxes/seg_{}.png".format(idx_dataset, idx_img), img_out)

                    # get traces from (not yet connected) boxes
                    traces = tra.calculateTraces()
  
                    # filter out traces that do not represent particles
                    traces_filt = TraceFilter(test_ds_raw).filter_traces(traces)
                    count_ds = len(traces_filt)
                    tDiff = time.time() - tBegin
                    print(f"found {count_ds} particle traces")
                    file.write("{:d},{},{:.2f}\n".format(count_ds, test_ds.root, tDiff))
                    file.flush()

main()