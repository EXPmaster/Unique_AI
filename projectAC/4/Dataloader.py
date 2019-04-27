import torchvision.datasets
import torch
from torch.utils.data import DataLoader
import torchvision.transforms

class Mydataset():
    def __init__(self, opt):
        self.train_set = torchvision.datasets.MNIST(root='./train/train-images.idx3-ubyte', train=True,
                                     transform = torchvision.transforms.ToTensor(),download=False)
        self.test_set = torchvision.datasets.MNIST(root='./train/train-images.idx3-ubyte', train=False,
                                    transform = torchvision.transforms.ToTensor(),download=False)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_set,
                                                        batch_size=opt.BATCH_SIZE,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_set,
                                                       batch_size=opt.BATCH_SIZE,
                                                       shuffle=False)
