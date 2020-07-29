# DL_CIFAR

Project for the Lecture "Deep Learning" held at Hasso-Plattner-Institute that is about applying the concept of Class Activation Mapping to the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) which contains 60,000 32x32 images of 10 different categories.

## Preview
In the following preview an overview of the influence of centercropping the input images before training is given as well as a first grasp of the impact of using pretrained models on the quality of the Class Activation Maps.
### Centercropping
|Original Image|Alexnet (no Centercropping)|Alexnet (Centercropping)|VGG19 (no Centercropping)|VGG19 (Centercropping)|
|-|-|-|-|-|
|![Dog Image](/images/Centercropping/dog.png)|![](/images/Centercropping/Alexnet_noCenterCrop_cam.png)|![](/images/Centercropping/Alexnet_CenterCrop_cam.png)|![](/images/Centercropping/VGG19_noCenterCrop_cam.png)|![](/images/Centercropping/VGG19_CenterCrop_cam.png)|
|![Car Image](/images/Centercropping/car.png)|![](/images/Centercropping/Alexnet_noCenterCrop_cam1.png)|![](/images/Centercropping/Alexnet_CenterCrop_cam1.png)|![](/images/Centercropping/VGG19_noCenterCrop_cam1.png)|![](/images/Centercropping/VGG19_CenterCrop_cam1.png)|

### Pretraining
|Original Image|Alexnet (not pretrained)|Alexnet (pretrained)|VGG19 (not pretrained)|VGG19 (pretrained)|
|-|-|-|-|-|
|![Car Image](/images/Pretraining/car.png)|![](/images/Pretraining/Alexnet_notPretrained_cam.png)|![](/images/Pretraining/Alexnet_Pretrained_cam.png)|![](/images/Pretraining/VGG19_notPretrained_cam.png)|![](/images/Pretraining/VGG19_Pretrained_cam.png)|
|![Horse Image](/images/Pretraining/horse.png)|![](/images/Pretraining/Alexnet_notPretrained_cam1.png)|![](/images/Pretraining/Alexnet_Pretrained_cam1.png)|![](/images/Pretraining/VGG19_notPretrained_cam1.png)|![](/images/Pretraining/VGG19_Pretrained_cam1.png)|

<!---
### Truck
![Truck Image](/images/truck.png)
![Truck CAM](/images/truck_cam.png)  
### Automobile
![Automobile Image](/images/automobile.png)
![Automobile CAM](/images/automobile_cam.png)
### Deer
![Deer Image](/images/deer.png)
![Deer CAM](/images/deer_cam.png)
-->
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
model = {vgg19, alexnet}  
output = output file name

### Model Training
- work in Progress

## Sources
- http://cnnlocalization.csail.mit.edu/
- https://medium.com/intelligentmachines/implementation-of-class-activation-map-cam-with-pytorch-c32f7e414923
- Main paper: http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf
