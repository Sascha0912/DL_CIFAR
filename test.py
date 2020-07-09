import torch
from d2l import torch as d2l
from data_loading import getDataLoaders
from architecture import AlexNet

def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    if not device:
        device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)  # num_corrected_examples, num_examples
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(d2l.accuracy(net(X), y), sum(y.shape))
    return metric[0] / metric[1]

def net_eval(dataloader, net, print_each=5):
    all     = 0
    correct = 0
    counter = 0
    print("Evaluating Network...")
    print("Size of Test Dataset: "+str(len(iter(dataloader))))
    for input, label in dataloader:
        pred = net(input)
        value, predLabel = torch.max(pred.data, 1)
        correct += (predLabel==label).sum().item()
        all += len(predLabel)
        counter += 1
        if (counter % print_each ==1):
            print("Evaluating "+str(counter))
    return correct, all

# Testing
'''
model_path = 'models/model.pth'
trainLoader, testLoader = getDataLoaders()
model = AlexNet()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

print(net_eval(testLoader, model))
'''
