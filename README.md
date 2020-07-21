# DL_CIFAR

Project for the Lecture "Deep Learning" held at Hasso-Plattner-Institute that is about applying the concept of Class Activation Mapping to the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) which contains 60,000 32x32 images of 10 different categories.

## Preview
On the left side we can find the original image as found in the dataset upscaled. On the right side we can find the corresponding image that has class activation mapping applied to it. The images shown here are chosen randomly to demonstrate the working of CAM.
### Truck
![Truck Image](/images/truck.png)
![Truck CAM](/images/truck_cam.png)  
### Automobile
![Automobile Image](/images/automobile.png)
![Automobile CAM](/images/automobile_cam.png)
### Deer
![Deer Image](/images/deer.png)
![Deer CAM](/images/deer_cam.png)

## File Structure

|File|Content|
|-|-|
|<code>data_loading.py</code>|Contains all functions that get the CIFAR10 dataset and preprocess it|
|<code>util.py</code>|Contains util functions that do not belong to a specific category|
|<code>config.py</code>|Contains parameters and the corresponding Getter Functions|
|<code>architecture.py</code>|This file contains the Class(es) representing the network structure of the neural network|
|<code>train.py</code>|Contains the main training function and the basic setup of the model|
|<code>test.py</code>|This file contains model evaluations functions AND all functionality regarding Class Activation mapping|

## Usage
### CAM visualization
- Currently to get a random image plus its CAM execute the following command:
```python
python test.py [-h] model output
```
model = {vgg16, alexnet}  
output = output file name

### Model Training
- work in Progress

## Sources
- http://cnnlocalization.csail.mit.edu/
- https://medium.com/intelligentmachines/implementation-of-class-activation-map-cam-with-pytorch-c32f7e414923
- Main paper: http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf
