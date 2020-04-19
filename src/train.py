import torch
from torch.utils.tensorboard import SummaryWriter

from src.data import train_loader

# Train on cpu or gpu if available
from src.helpers import plot_classes_preds


def train(criterion, optimizer, n_epochs, net, device, writer):

    writer.add_graph(net, next(iter(train_loader))[0])

    running_loss = 0.0
    n_correct = 0
    # An epoch is a complete pass thru a dataset
    for epoch in range(n_epochs):

        # Transfer model to cpu / gpu -- whichever is available
        net.to(device)

        # Get a batch of data, use enumerate to iterate
        # over the trainloader; each iteration in the for-loop
        # gives us a batch of images and their labels and a counter `i`
        # which starts at 0
        for i, data in enumerate(train_loader, start=0):
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

            running_loss += loss.item()

            # Make a matrix of the predictions
            _, predicted = torch.max(output.data, 1)
            n_correct += (predicted == labels).sum().item()
            if i % 100 == 99:  # i.e. every hundred mini-batches

                # calulate mean vals and add to tensorboard
                writer.add_scalar('Training/Loss', running_loss / 100, epoch *
                                  len(train_loader) + i)

                writer.add_scalar('Training/accuracy', n_correct / 100,
                                  epoch *
                                  len(train_loader) + i)

                writer.close()
                running_loss = 0.0
                n_correct = 0

    print('Finished training')
