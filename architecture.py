import torch
import torch.nn as nn
from torchvision.models import alexnet, vgg19

'''
# this network structure is added to a pretrained vgg16
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc=nn.Linear(512,10, bias = False)

    def forward(self,x):
        dim = x.shape[0]
        v=x.view(dim,512,-1)
        x=v.mean(2)
        x=x.view(1,dim,512)
        x=self.fc(x)

        return x.view(-1,10)

# this network structure is added to a pretrained alexnet
class Net2(nn.Module):
    def __init__(self):
        super(Net2,self).__init__()
        self.fc=nn.Linear(256,10, bias = False)

    def forward(self,x):
        dim = x.shape[0]
        v=x.view(dim,256,-1)
        x=v.mean(2)
        x=x.view(1,dim,256)
        x=self.fc(x)

        return x.view(-1,10)
'''

# VGG19: Test accuracy: 92%
class Vgg19(nn.Module):
    def __init__(self, pretrained = False, freeze = None):
        super().__init__()

        # get the pretrained Vgg19_bn network
        vgg19_net = vgg19(pretrained = pretrained)

        # disect the network to access its last convolutional layer
        self.features = vgg19_net.features

        # get the max pool of the features stem
        self.avg_pool = vgg19_net.avgpool

        # get the classifier of the vgg19_bn
          # Standard adapted to Outputsize 10
          # self.classifier = vgg19_net.classifier
          # self.classifier[3] = nn.Linear(4096,1024)
          # self.classifier[6] = nn.Linear(1024,10)
        # Single Classifier Layer for CAM
        self.classifier = nn.Linear(512,10)

        # placeholder for the gradients
        self.gradients = None

        if(freeze):
          for i, param in enumerate(self.features.parameters()):
            param.requires_grad = False
            if(i+1 >= freeze):
              break

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features(x)
        # register the hook
        # h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        # x = self.avg_pool(x)
        # x = x.view(-1, 25088)
        # x = self.classifier(x)
        # return x

        x = self.avg_pool(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, 512, -1)
        x = x.mean(2)
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features(x)


# Alexnet (Freeze=12) Test accuracy: 69%
class Alexnet(nn.Module):
    def __init__(self, pretrained = False, freeze = None):
        super().__init__()

        # get the pretrained AlexNet network
        alex_net = alexnet(pretrained = pretrained)

        # disect the network to access its last convolutional layer
        self.features = alex_net.features

        # get the max pool of the features stem
        self.avg_pool = alex_net.avgpool

        # get the classifier of the AlexNet
          # Standard adapted to outputsize 10
          # self.classifier = alex_net.classifier
          # self.classifier[4] = nn.Linear(4096,1024)
          # self.classifier[6] = nn.Linear(1024,10)
        # Single Classifier Layer for CAM
        self.classifier = nn.Linear(256, 10)

        # placeholder for the gradients
        self.gradients = None

        if(freeze):
          for i, param in enumerate(self.features.parameters()):
            param.requires_grad = False
            if(i+1 >= freeze):
              break


    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features(x)
        # register the hook     (Only GRAD_CAM ?!)
        # h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.avg_pool(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, 256, -1)
        x = x.mean(2)
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features(x)
