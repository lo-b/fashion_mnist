import torch
from torch import optim
import torch.nn as nn
from src.model import Net
from src.test import test

# Create the network and print it's architecture
from src.train import train
from src.visualize_tensorboard import write_images, write_embedding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net()
print(net)

# Visualize data in TensorBoard
write_images()

# TODO: check if below works on desktop
# Line below doesn't work due to WebGL problems on laptop
# write_embedding()

# Specify the loss function; for a classification
# problem with classes a `cross entropy loss`
# works well.
criterion = nn.CrossEntropyLoss()

# Define learning rate `alpha`
alpha = 0.01

# Specify an optimizer which will do backpropagation
# for us using autograd (a module that keeps track of a tensor's lifecycle)
optimizer = optim.SGD(net.parameters(), lr=alpha, momentum=0.9)

train(criterion, optimizer, 5, net, device)
test(net, device)

