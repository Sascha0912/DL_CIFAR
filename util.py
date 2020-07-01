import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',' truck')

def imshow(image):
    np_image = np.transpose(image.numpy(), (1,2,0))
    plt.figure(figsize=(2,2))
    plt.imshow(np_image)
    plt.show()

# move to test.py
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    if not device:
        device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)  # num_corrected_examples, num_examples
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(d2l.accuracy(net(X), y), sum(y.shape))
    return metric[0] / metric[1]
