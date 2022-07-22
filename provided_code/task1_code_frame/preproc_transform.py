import torch

class PreprocessingTransform(object):
    """
    A transform to calculate a preprocessed image from a block of images.
    """
    def __init__(self, device=None):
        """
        device: device to execute calculations on (using the original device where 
                    the data is located on before the transform takes place)
        """
        self.device = device


    def __call__(self, sample):
        """
        input:  
            sample: a block of frames of length w
                        shape: (1, w, X, Y)

            output: a single image that represents the change over time for each pixel
                        shape: (1, X, Y)
        """
        if self.device is not None:
            sample = sample.to(self.device)

        w = sample.shape[1]//2
        q1 = torch.mean(sample[:, :w, :, : ], dim=1)
        q2 = torch.mean(sample[:, w:, :, : ], dim=1)
        img = q2 - q1
        img = img.view(1, 1, img.shape[-2], img.shape[-1])
        return img