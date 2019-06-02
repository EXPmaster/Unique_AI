import pickle
import numpy as np
import os
import torch
from PIL import Image
cifar_file = './cifar'

# Cifar10
def read_single(file):
    with open(file, 'rb') as f:
        dataset = pickle.load(f, encoding='bytes')
        sample = dataset[b'data']
        label = dataset[b'labels']
        sample = sample.reshape([10000, 3, 32, 32]).transpose([0, 2, 3, 1])
        return sample, label


def convert_image(X, imsize=10000):
    print('converting image...')
    image_resized = np.zeros((imsize, 100, 100, 3)) # 227
    for i in range(imsize):
        img = X[i]
        img = Image.fromarray(img)
        img = np.array(img.resize((100, 100), Image.BICUBIC))
        image_resized[i, :, :, :] = img
    image_resized /= 255
    image_resized = image_resized.transpose([0, 3, 1, 2])
    print('convert finished...')
    return image_resized


def readfile(path=cifar_file):
    dataset = []
    labelset = []
    for i in range(1, 3):
        file = os.path.join(path, 'data_batch_%d' % i)
        x, y = read_single(file)
        dataset.append(x)
        labelset.append(y)
    trainset = np.concatenate(dataset)
    trainset = convert_image(trainset, len(trainset))
    trainlabel = np.concatenate(labelset)
    testfile = os.path.join(path, 'test_batch')
    testset, testlabel = read_single(testfile)
    testset = convert_image(testset)
    # trainlabel = np.eye(10)[trainlabel]
    # testlabel = np.eye(10)[testlabel]

    trainset = torch.Tensor(trainset)
    trainlabel = torch.Tensor(trainlabel).long()
    testset = torch.Tensor(testset)
    testlabel = torch.Tensor(testlabel).long()

    return trainset, trainlabel, testset, testlabel


