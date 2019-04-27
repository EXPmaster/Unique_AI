import torch.backends.cudnn as cudnn
from Dataloader import Mydataset
import torch
import torch.nn as nn
from Config import config
import sys
import os
sys.path.append(os.getcwd())


opt = config()


class MyNetwork(nn.Module):
    def __init__(self, opt):
        super(MyNetwork, self).__init__()
        self.f1 = nn.Sequential(
            nn.Linear(opt.INPUT_SIZE, opt.HIDDEN_SIZE),
            nn.BatchNorm1d(opt.HIDDEN_SIZE),  # 正则化
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合，随机关闭0.5的神经元
            nn.Linear(opt.HIDDEN_SIZE, opt.NUM_CLASS)
        )

    def forward(self, x):
        outputs = self.f1(x)
        return outputs


dataset = Mydataset(opt)
net = MyNetwork(opt)
criterion = nn.CrossEntropyLoss()

net = torch.nn.DataParallel(
    net, device_ids=range(torch.cuda.device_count())).cuda()
cudnn.benchmark = True


def train(epoch):
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.LR)
    print('train epoch %d' % (epoch + 1))
    for iteration, (inputs, labels) in enumerate(dataset.train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs = torch.autograd.Variable(inputs.view(-1, 28 * 28))
        labels = torch.autograd.Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if iteration % 600 == 0:
            print('current loss = %.2f' % loss.item())
    print('train epoch %d finish' % (epoch + 1))
    print('saving model...')
    torch.save(net, opt.MODEL_PATH)
    print('model saved')


def test_model():
    net.eval()
    total = 0
    currect = 0
    for inputs, labels in dataset.test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs = torch.autograd.Variable(inputs.view(-1, 28 * 28))
        outputs = model(inputs)
        _, predicts = torch.max(outputs.data, 1)
        total += labels.size(0)
        currect += (predicts == labels).sum()
    print('accuracy: %.2f' % (100 * currect / total))


if __name__ == '__main__':
    for i in range(opt.EPOCH):
        train(i)

    print('loading model...')
    model = torch.load(opt.MODEL_PATH)
    model.eval()
    test_model()
