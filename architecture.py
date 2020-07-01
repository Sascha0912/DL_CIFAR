import torch
import torch.nn as nn
from data_loading import getDataLoaders

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
        print(features)
        flatten  = self.avgPooling(features).view(features.size(0), -1)
        print(flatten)
        output   = self.classifier(flatten)
        print(output)
        return output, features

class KaggleCNN(nn.Module):
    def __init__(self):
        super(KaggleCNN, self).__init__()
        self.net = nn.Sequential(
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

# TESTING - to verify network logic #
'''
cnn = KaggleCNN()
trainLoader, testLoader = getDataLoaders()

trainIter = iter(trainLoader)
images, labels = trainIter.next()

print(images[0].shape)

X = torch.randn(size=(1, 3, 32, 32), dtype=torch.float32)
for layer in cnn.net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
'''
