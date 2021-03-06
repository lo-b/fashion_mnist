"""
This code is taken from the PyTorch tutorial:
`Visualizing Models, Data, and Training with TensorBoard`
https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# helper function to show an image
# (used in the `plot_classes_preds` function below)
from src.data import classes


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def select_n_random(data, labels, n=100):
    """
    Selects n random data points and their corresponding labels from a dataset
    """
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    net.to('cpu')
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in
                   zip(preds, output)]


def plot_classes_preds(net, images, labels):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    net.to('cpu')
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig
