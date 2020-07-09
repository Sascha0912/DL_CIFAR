import torch
import torch.nn as nn
from data_loading import getDataLoaders
from util import imshow

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1,3,32,32)

class ExampleCNN(nn.Module):
    def __init__(self):
        super(ExampleCNN, self).__init__()
        self.net = nn.Sequential(
            Reshape(),
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 10, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
        self.avgPooling = nn.AvgPool2d(kernel_size=4)
        self.classifier = nn.Linear(4, 40)

    def forward(self, x):
        features = self.net(x)
        flatten  = self.avgPooling(features).view(features.size(0), -1)
        output   = self.classifier(flatten)
        return output, features

class KaggleCNN(nn.Module):
    def __init__(self):
        super(KaggleCNN, self).__init__()
        self.features = nn.Sequential(
            Reshape(),
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            Reshape(),
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.linear = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 10),
        )

    def forward(self, x):
        features = self.features(x)
        flatten = features.view(features.size(0), -1)
        lin = self.linear(flatten)
        return lin

# TESTING - to verify network logic #
'''
cnn = AlexNet()
trainLoader, testLoader = getDataLoaders()

testIter = iter(testLoader)
images, labels = testIter.next()

imshow(images[0])
'''
