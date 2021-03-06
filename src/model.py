import torch
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):

    # In the constructor the
    # layers are defined
    def __init__(self):
        super(Net, self).__init__()

        # The first layer is a convolution layer with 1 input channel (since
        # the photos are in grayscale) and 10 output channels / feature maps (
        # by applying 10 image filters). The image filters are convolution
        # kernels with shape (3,3)
        #
        # The general formula for the output of the layers is
        # output_dim = (W-F)/S + 1 where `W` is the image's height || width,
        # `F` the filter size and `S` the stride.
        #
        # Using the formula above:
        # (28-3)/1 + 1 = 25/1 + 1 = 26
        # so it's output shape will be (batch_size, 10, 26, 26)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, stride=1, padding=1)

        # The second layer is a MaxPooling layer. With a kernel of shape (2,2)
        # and a stride of 2. The MaxPooling layer will convolve with it's
        # input and for each grid it select's the maximum value -- this
        # basically allows the network to abstract features that are
        # present, no matter the orientation or size of an object.
        #
        # Using the formula above:
        # (26-2)/2 + 1 = 24/2 + 1 = 13
        # so it's output shape will be (batch_size, 10, 13, 13)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        # (13-3)/1 + 1 = 11
        # Hence the output of a single image will be in shape (32, 11, 11)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # Before the fully connected layer another MaxPooling layer is
        # applied again.
        #
        # (11-2)/2 + 1 = 9/2 + 1 = 5.5 -- rounded down this is 5;
        # So the shape of a single image is (32, 5, 5)
        # A fully connected layer to all our classes (which there were 10 of)
        self.fc1 = nn.Linear(in_features=7 * 7 * 64, out_features=10)

    # Feed forward behaviour of the network
    def forward(self, x):
        # Feed it thru the convolutional layer
        x = self.conv1(x)

        # Apply a ReLU to it
        x = F.relu(x)

        # print('Shape after CONV1:', x.shape)

        # Feed it thru the MaxPooling layer
        # It's output will be in shape (batch_size, 10, 13, 13)
        x = self.maxpool1(x)

        # print('Shape after MAXPOOL1:', x.shape)

        x = self.conv2(x)

        x = F.relu(x)

        # print('Shape after CONV2:', x.shape)

        # We will use the same maxpool layer used after conv1
        x = self.maxpool1(x)

        # print('Shape after MAXPOOL2:', x.shape)

        # Flatten the output from the MaxPooling layer
        # so it can be fed into the fully connected layer.
        # This transforms (batch_size, 10, 13, 13) into
        # (batch_size, 13 * 13 * 10)
        x = x.view(x.size(0), -1)

        # print('Shape after RESHAPE:', x.shape)

        # Pass it thru the fully connected layer.
        x = self.fc1(x)

        # The last nodes will contain float values; we for these
        # to be probabilities so we use a softmax function that will
        # transform our floats to probabilities -- all added up being 1.
        #
        # Since our first dimension is our batch size we want the softmax
        # to be performed over the second dimension `1` which will be the
        # values themselves.
        x = F.log_softmax(x, dim=1)

        return x
