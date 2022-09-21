import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = down_block(1, 8)
        self.down2 = down_block(8, 16)
        self.down3 = down_block(16, 32)
        self.down4 = down_block(32, 64)

        self.fc1 = nn.Linear(in_features=64, out_features=16)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=16, out_features=1)
        
    def forward(self, x):
            x = self.down1(x)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)

            # global pooling
            # can also be found at https://github.com/Konnsy/REAML2022-hackathon/wiki/Useful-code-snippets
            x = torch.max(x.view(*x.shape[:2], -1), dim=2)[0]

            x = self.fc1(x)
            x = self.act1(x)
            x = self.fc2(x)
            x = torch.sigmoid(x)
            return x

def down_block(c_in, c_out):
    """
    Creates two connected convolutional layers followed by a 2x2 downsampling.
        param:
            c_in: size of the input channels.
            c_out: the desired amount of output channels

    """
    return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, dilation=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(c_out, c_out, 3, padding=1, dilation=1, padding_mode='zeros'),
            nn.MaxPool2d(2)
            )