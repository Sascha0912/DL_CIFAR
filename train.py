import torch
from torch import nn
from torchvision import models
from config import getLearningRate
from architecture import Net
from data_loading import getDataLoader

# Getting CUDA information
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

# Get pretrained network and remove last layer
net = models.vgg16(pretrained=True)
mod = nn.Sequential(*list(net.children())[:-1])

# Add custom Net class to model
model=nn.Sequential(mod,Net())
print("Model structure initialized.")

def getModel():
    return model
def getMod():
    return mod

trainable_parameters = []
for name, p in model.named_parameters():
    if "fc" in name:
        trainable_parameters.append(p)

optimizer = torch.optim.SGD(params=trainable_parameters, lr=getLearningRate(), momentum=1e-5)
criterion = nn.CrossEntropyLoss()

train_loader = getDataLoader()
total_step = len(train_loader)
loss_list = []
acc_list = []

def train(model_name, use_gpu=False):
    if (use_gpu):
        model.to(device)
    min_loss=9999
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if (use_gpu):
                images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i % 100) == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

        with open('loss.txt', 'w+') as f:
            f.write("%s\n" % loss)
        if loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), model_name)
