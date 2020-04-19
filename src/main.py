import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from src.model import Net
from src.test import test
# Create the network and print it's architecture
from src.train import train
from src.visualize_tensorboard import write_images

writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net()
print(net)

# Visualize data in TensorBoard
write_images(writer)

# TODO: check if below works on desktop
# Line below doesn't work due to WebGL problems on laptop
# write_embedding()

# Specify the loss function; for a classification
# problem with classes a `cross entropy loss`
# works well.
criterion = nn.CrossEntropyLoss()

# Define learning rate `alpha`
alpha = 0.01

# Number of epochs
n_epochs = 2

# Specify an optimizer which will do backpropagation
# for us using autograd (a module that keeps track of a tensor's lifecycle)
optimizer = optim.SGD(net.parameters(), lr=alpha, momentum=0.9)

train(criterion, optimizer, n_epochs, net, device, writer)
test(net, device, writer)
