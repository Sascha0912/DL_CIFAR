import torch
import torch.nn as nn
from torchvision.models import alexnet, vgg19

class Vgg19(nn.Module):
    def __init__(self, pretrained = False, freeze = None):
        super().__init__()

        # get the pretrained Vgg19_bn network
        vgg19_net = vgg19(pretrained = pretrained)

        # disect the network to access its last convolutional layer
        self.features = vgg19_net.features

        # get the max pool of the features stem
        self.avg_pool = vgg19_net.avgpool

        # Single Classifier Layer for CAM
        self.classifier = nn.Linear(512,10)

        if(freeze):
          for i, param in enumerate(self.features.parameters()):
            param.requires_grad = False
            if(i+1 >= freeze):
              break

    def forward(self, x):
        x = self.features(x)

        x = self.avg_pool(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, 512, -1)
        x = x.mean(2)
        x = self.classifier(x)
        return x

class Alexnet(nn.Module):
    def __init__(self, pretrained = False, freeze = None):
        super().__init__()

        # get the pretrained AlexNet network
        alex_net = alexnet(pretrained = pretrained)

        # disect the network to access its last convolutional layer
        self.features = alex_net.features

        # get the max pool of the features stem
        self.avg_pool = alex_net.avgpool

        # Single Classifier Layer for CAM
        self.classifier = nn.Linear(256, 10)

        if(freeze):
          for i, param in enumerate(self.features.parameters()):
            param.requires_grad = False
            if(i+1 >= freeze):
              break

    def forward(self, x):
        x = self.features(x)

        # apply the remaining pooling
        x = self.avg_pool(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, 256, -1)
        x = x.mean(2)
        x = self.classifier(x)
        return x
