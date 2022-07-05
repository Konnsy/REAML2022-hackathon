import torch

class PreprocessingTransform(object):
    """
    A transform to help modify the input data
    input:  a Block of images(frames)
                shape: (1, BlockSize, X, Y)
    output: a single image that represents the change over time for each pixel
                shape: (1, X, Y)
    """
    def __call__(self, sample):        
        w = sample.shape[1]//2
        q1 = torch.mean(sample[:, :w, :, : ], dim=1)
        q2 = torch.mean(sample[:, w:, :, : ], dim=1)
        img = q2 - q1
        img = img.view(1, 1, img.shape[-2], img.shape[-1])
        return img