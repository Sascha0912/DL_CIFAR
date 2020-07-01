import torch
import torchvision
import torchvision.transforms as transforms

# CONSTANTS
DATA_PATH = "./data"
BATCH_SIZE = 64

def getDataLoaders():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train = torchvision.datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=transform)
    test  = torchvision.datasets.CIFAR10(DATA_PATH, train=False, download=True, transform=transform)

    trainLoader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testLoader  = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return trainLoader, testLoader
