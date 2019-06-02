from AlexNet import AlexNet
from DataLoader import *
import torch.utils.data
import torch.optim
import torch.nn as nn
import torch.autograd
import torch.backends.cudnn as cudnn

print('Loading...')
trainset, trainlabel, testset, testlabel = readfile()
new_trainset = torch.utils.data.TensorDataset(trainset, trainlabel)
new_testset = torch.utils.data.TensorDataset(testset, testlabel)
trainloader = torch.utils.data.DataLoader(dataset=new_trainset, batch_size=16, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=new_testset, batch_size=16, shuffle=False)
print('Load finished')

cudnn.benchmark = True
net = AlexNet()
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

net.to(device)
criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)


def train(epoch):
    net.train()
    print('epoch %d' % (epoch + 1))

    total = correct = 0
    for i, (inputs, labels) in enumerate(trainloader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predicts = torch.argmax(outputs.data, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        # labels = [torch.argmax(label).numpy() for label in labels]
        # labels = np.array(labels)

        total += len(labels)
        correct += (predicts == labels).sum()

        if i % 600 == 0:
            print('current loss = %lf' % loss.item())
    print('train acc: %lf' % (100 * correct / total))


# @torch.no_grad()
def valid():
    print('validating...')
    net.eval()
    total = 0
    correct = 0
    for inputs, labels in testloader:
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(inputs)

        predicts = torch.argmax(outputs.data, 1).cpu().numpy()
        labels = labels.cpu().numpy()
        # labels = [torch.argmax(label).numpy() for label in labels]
        # labels = np.array(labels)

        total += len(labels)
        correct += (predicts == labels).sum()
    print('valid accuracy: %lf' % (100 * correct / total))


def main():
    epoch = 20
    print('Start training...')
    for i in range(epoch):
        train(i)
        valid()

    print('Train finished')


if __name__ == '__main__':
    main()