# initialize tensor and lists to monitor test loss and accuracy
import numpy as np
import torch

from src.data import test_loader, batch_size, classes


def test_udacity(net, criterion, device):
    test_loss = torch.zeros(1)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # set the module to evaluation mode
    net.eval()

    for batch_i, data in enumerate(test_loader):

        # get the input images and their corresponding labels
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward pass to get outputs
        outputs = net(inputs)

        # calculate the loss
        loss = criterion(outputs, labels)

        # update average test loss
        test_loss = test_loss + (
                (torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))

        # get the predicted class from the maximum value in the output-list of class scores
        _, predicted = torch.max(outputs.data, 1)

        # compare predictions to true label
        # this creates a `correct` Tensor that holds the number of correctly classified images in a batch
        correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))

        # calculate test accuracy for *each* object class
        # we get the scalar value of correct items for a class, by calling `correct[i].item()`
        for i in range(batch_size):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print(
                'Test Accuracy of %5s: N/A (no training examples)' % (
                    classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
