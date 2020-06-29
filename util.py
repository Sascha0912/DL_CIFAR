import numpy as np
import matplotlib.pyplot as plt

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',' truck')

def imshow(image):
    np_image = np.transpose(image.numpy(), (1,2,0))
    plt.figure(figsize=(2,2))
    plt.imshow(np_image)
    plt.show()
