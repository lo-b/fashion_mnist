# Import modules for loading & transforming data --
# usually images are also normalized but this is
# already done when we use the dataset-class FashionMNIST.
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

# Transforms-module, usually this is used for pre-processing
# but we only use it for transforming our data to tensors.
from torchvision import transforms

# Images from the FashionMNIST dataset are PIL-images
# having values between [0,1]. We transform these
# (probably a numpy array) to a PyTorch tensor.
data_transform = transforms.ToTensor()
train_data = FashionMNIST(root='../data', download=True,
                          transform=data_transform)
test_data = FashionMNIST(root='../data', train=False, download=True,
                         transform=data_transform)

batch_size = 20

# Some stats about our data
print('Number of train images:', len(train_data))
print('Number of test images:', len(test_data))


# Make DataLoaders where we define our batch size
# shuffle before each retrieval of a batch.
train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size,
                         shuffle=True)

# The classes: each image belongs to one and only one
# of these classes.
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
