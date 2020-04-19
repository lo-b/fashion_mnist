import torch
from torch.utils.tensorboard import SummaryWriter

from src.data import train_loader

# Train on cpu or gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(criterion, optimizer, n_epochs, net):

    writer = SummaryWriter()
    writer.add_graph(net, next(iter(train_loader))[0])
    count = 0
    # An epoch is a complete pass thru a dataset
    for epoch in range(n_epochs):

        # Transfer model to cpu / gpu -- whichever is available
        net.to(device)

        # Get a batch of data
        for batch_index, data in enumerate(train_loader):
            images, labels = data

            # transfer images and labels to device
            images, labels = images.to(device), labels.to(device)

            # The optimizer is accumalitive so we need to reset
            # it after each backprop step
            optimizer.zero_grad()

            # Do a forward pass thru the network
            output = net.forward(images)

            # Calculate the loss
            loss = criterion(output, labels)

            # Backprop to calculate the gradient
            loss.backward()

            # Update the gradients
            optimizer.step()

            writer.add_scalar('Loss/Train', loss, count)

            count += 1

    print('Finished training')
