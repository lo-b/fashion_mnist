"""
This module uses TensorBoard to visualize our data and to visualize
our model so we can analyse it later.
"""

import torchvision
from torch.utils.tensorboard import SummaryWriter

# import DataLoader
from src.data import train_loader

# get a batch of images
images, labels = next(iter(train_loader))


print('shape images:', images.shape)
# By default the writer will output to ./runs/
writer = SummaryWriter()

grid = torchvision.utils.make_grid(images)

writer.add_image('Batch of train images', grid, 0)
writer.close()

