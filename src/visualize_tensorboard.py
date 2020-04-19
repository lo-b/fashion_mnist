"""
This module uses TensorBoard to visualize our data and to visualize
our model so we can analyse it later.
"""

import torchvision
from torch.utils.tensorboard import SummaryWriter

from src.data import train_data, classes
from src.helpers import select_n_random

# Writer object to interact with TensorBoard
writer = SummaryWriter()


def write_images():
    # import DataLoader
    from src.data import train_loader

    # get a batch of images
    images, labels = next(iter(train_loader))

    print('shape images:', images.shape)
    # By default the writer will output to ./runs/

    grid = torchvision.utils.make_grid(images)

    writer.add_image('Batch of train images', grid, 0)
    writer.close()


def write_embedding():
    """
    Visualize dataset in higher 3D space for exploration
    """
    # select random images and their target indices
    images, labels = select_n_random(train_data.data, train_data.targets)

    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]

    # log embeddings
    features = images.view(-1, 28 * 28)
    writer.add_embedding(features,
                         metadata=class_labels,
                         label_img=images.unsqueeze(1))
    writer.close()
    print("Write embedding done...")
