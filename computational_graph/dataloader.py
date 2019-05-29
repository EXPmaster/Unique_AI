import torchvision.datasets
import torch
from torch.utils.data import DataLoader
import torchvision.transforms


class Mydataset():
    def __init__(self):
        self.train_set = torchvision.datasets.MNIST(
            root='./train/train-images.idx3-ubyte',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=False)
        self.test_set = torchvision.datasets.MNIST(
            root='./train/train-images.idx3-ubyte',
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=False)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_set,
                                                        batch_size=100,
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_set,
                                                       batch_size=100,
                                                       shuffle=False)


