import torch
import torch.nn as nn


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input size: (18, 1000, 1000)
        self.pool1 = nn.MaxPool2d(4, 4)
        # Pool1 output size: (18, 250, 250)
        self.conv1 = nn.Conv2d(18, 3, (5, 5))
        # Conv1 output size: (3, 246, 246)
        self.conv2 = nn.Conv2d(3, 1, (5, 5))
        # Conv2 output size: (3, 242, 242)
        self.pool2 = nn.MaxPool2d(4, 4)
        # Pool2 output size: (3, 60, 60)
        self.fc1 = nn.Linear(3*60*60, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.pool1(x)
        x = self.conv1(x)
        x = nn.ReLU(x)
        x = self.conv2(x)
        x = nn.ReLU(x)
        x = self.pool2(x)
        x = self.fc3(self.fc2(self.fc1(x)))

