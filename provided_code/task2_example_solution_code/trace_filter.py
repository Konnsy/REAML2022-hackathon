import math
import torch
import torch.nn as nn
import sys
import os
import os.path as osp

class TraceFilter:
    def __init__(self, raw_dataset):
        # optional TODO: implement
        pass

    def filter_traces(self, traces):
        return list(filter(lambda t : self.is_particle(t), traces))

    def is_particle(self, trace):
        # optional TODO: implement 
        return True
            