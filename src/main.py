from torch import optim
import torch.nn as nn
from src.model import Net
from src.data import test_loader

# Create the network and print it's architecture
from src.train import train

net = Net()
print(net)

# Specify the loss function; for a classification
# problem with classes a `cross entropy loss`
# works well.
criterion = nn.CrossEntropyLoss()

# Define learning rate `alpha`
alpha = 0.01

# Specify an optimizer which will do backpropagation
# for us using autograd (a module that keeps track of a tensor's lifecycle)
optimizer = optim.SGD(net.parameters(), lr=alpha)

train(criterion, optimizer, 10, net)
