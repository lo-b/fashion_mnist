import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from src.model import Net
from src.test import test
from src.test_udacity import test_udacity
from src.train import train
from src.visualize_tensorboard import write_images, visualize_test_results

if __name__ == '__main__':
    # Define learning rate `alpha`
    alpha = 0.001

    # Number of epochs
    n_epochs = 5
    # Define momentum
    momentum = 0.0

    comment = 'n_epochs={}_alpha={:.3f}_momentum={:.3f}_adam'.format(n_epochs,
                                                                     alpha,
                                                                     momentum)

    writer = SummaryWriter(flush_secs=1, comment=comment)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    net = Net()
    print(net)

    # Visualize data in TensorBoard
    write_images(writer)

    # Specify the loss function; for a classification
    # problem with classes a `cross entropy loss`
    # works well.
    criterion = nn.NLLLoss()

    # Specify an optimizer which will do backpropagation
    # for us using autograd (a module that keeps track of a tensor's lifecycle)
    optimizer = optim.Adam(net.parameters(), lr=alpha)

    train(criterion, optimizer, n_epochs, net, device, writer)
    test(net, device, writer)
    test_udacity(net, criterion, device)

    PATH = 'checkpoints/'
    # The models state dictionary are key value pairs for the layer's
    # learned parameters (weight / biases) to a tensor
    torch.save(net.state_dict(), PATH + 'checkpoint_1.pt')

    visualize_test_results(device)
