This code is an example solution of task 2 and is implementing the code in "task2_code_frame".

changes to the code frame:
• The segmentation is implemented as a Resnet-like network of 100 layers 
  with 8 filters per convolution operation. The training process is implemented in the same file.
• The training section caches the preprocessed files so that preprocessing has to be done only
  once per file instead of having to do that in each epoch (this is just a speed-up).
• An augmentation transform to change the size and contrast of training images randomly is added.
• The orignal preprocessing is used without modifications.
• There are sections to write visualizations of the validation and test runs in the example code.
  Comment them in if you want to see the detections by eye instead of only looking at numbers.

Please note that the given code is just an example solution and that higher accuracies are possible
with further modifications.