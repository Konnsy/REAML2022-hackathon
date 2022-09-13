import torch
import torch.nn as nn
import statistics
import random
import copy

class Segmenter(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
          
    def forward(self, x):
        # TODO
        return x

    def train_model(self, ds_train, ds_val, device):
        # TODO
        pass      

    def save(self, model_path):
        torch.save(copy.deepcopy(self).cpu().state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
