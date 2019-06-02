import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            # stride 1
            nn.Conv2d(3, 96, 11, 4),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        self.max_pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.max_pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU()

        )
        self.max_pool3 = nn.MaxPool2d(3, 2)

        self.fc = nn.Sequential(
            nn.Linear(1024, 4096),  # 4096
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.max_pool1(x)
        # print(x.shape)

        x = self.conv2(x)
        # print(x.shape)
        x = self.max_pool2(x)
        # print(x.shape)

        x = self.conv3(x)
        # print(x.shape)
        x = self.max_pool3(x)
        # print(x.shape)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
