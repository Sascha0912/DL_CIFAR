import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l

def getClasses():
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',' truck')
    return classes

def getAlexNetClasses():
    classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    return classes

def imshow(image):
    np_image = np.transpose(image.numpy(), (1,2,0))
    plt.figure(figsize=(2,2))
    plt.imshow(np_image)
    plt.show()
