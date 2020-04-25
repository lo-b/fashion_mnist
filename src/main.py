import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from src.model import Net
from src.test import test
from src.test_udacity import test_udacity
from src.train import train
from src.visualize_tensorboard import write_images

# Define learning rate `alpha`
alpha = 0.001

# Number of epochs
n_epochs = 10

# Define momentum
momentum = 0.9

comment = 'n_epochs={}_alpha={:.3f}_momentum={:.3f}_adam'.format(n_epochs,
                                                                alpha,
                                                            momentum)

writer = SummaryWriter(comment=comment)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

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
criterion = nn.NLLLoss()


# Specify an optimizer which will do backpropagation
# for us using autograd (a module that keeps track of a tensor's lifecycle)
optimizer = optim.Adam(net.parameters(), lr=alpha, momentum=momentum)

train(criterion, optimizer, n_epochs, net, device, writer)
test(net, device, writer)
test_udacity(net, criterion, device)
