import torch
import torch.nn as nn

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
