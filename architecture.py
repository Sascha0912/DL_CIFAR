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
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 10, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
        self.avgPooling = nn.AvgPool2d(kernel_size=4)
        self.classifier = nn.Linear(10, 10)

    def forward(self, x):
        features = self.net(x)
        flatten  = self.avgPooling(features).view(features.size(0), -1)
        output   = self.classifier(flatten)
        return output, features

# TESTING - to verify network logic #
'''
cnn = ExampleCNN()
trainLoader, testLoader = getDataLoaders()

trainIter = iter(trainLoader)
images, labels = trainIter.next()

print(images[0].shape)

X = torch.randn(size=(1, 3, 32, 32), dtype=torch.float32)
for layer in cnn.net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
'''
