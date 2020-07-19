import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from config import getBatchSize

# CONSTANTS
DATA_PATH = "./data"

def getDataLoader():
    img_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

    train_data_dir = datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=img_transform)
    train_size = int(0.8 * len(train_data_dir))
    test_size = len(train_data_dir) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_data_dir, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=getBatchSize(), shuffle=True)

    return train_loader

def preprocessTestImages():
    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       normalize
    ])
    return preprocess

# get a test loader with batch size 1 (to visualize example images)
def getTestLoader():
    test_dataset = datasets.CIFAR10("./testdata", train=False, download=True, transform=preprocessTestImages())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    return test_loader
