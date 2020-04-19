import torch

from src.data import test_loader
import torch.nn.functional as F


def test(net, device, writer):
    class_labels = []
    class_preds = []

    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            # max return (values, index) here we're interested
            # in the indexes
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            probs = [F.softmax(output) for output in outputs]
            class_preds.append(probs)
            class_labels.append(predicted)

        # for each class stack predictions
        class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
        class_labels = torch.cat(class_labels)

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy: {acc} %')

        classes = range(10)
        for i in classes:
            labels_i = class_labels == i
            preds_i = class_preds[:, i]
            writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
            writer.close()
