"""
This module uses TensorBoard to visualize our data and to visualize
our model so we can analyse it later.
"""
import torch
import numpy as np

from src.data import train_data, classes, test_loader, train_loader
from src.helpers import select_n_random
import matplotlib.pyplot as plt
import torchvision

from src.model import Net


def write_images(writer):
    # get a batch of images
    images, labels = next(iter(train_loader))

    print('shape images:', images.shape)
    # By default the writer will output to ./runs/

    grid = torchvision.utils.make_grid(images)

    writer.add_image('Batch of train images', grid, 0)
    writer.close()


def write_embedding(writer):
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


def visualize_test_results(device):
    net = Net()
    net.load_state_dict(torch.load('checkpoints/checkpoint_1.pt'))
    net.to(device)
    net.eval()

    images, labels = next(iter(test_loader))

    images = images.to(device)

    probs = net.forward(images)


    # get predictions
    preds = np.squeeze(probs.data.max(1, keepdim=True)[1].cpu().numpy())
    images = images.cpu().numpy()

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        ax.set_title(
            "{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx] else "red"))

    fig.savefig('myfig.png')
