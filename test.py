import torch
import numpy as np
from data_loading import getTestLoader
from train import getModel
from train import getMod
from architecture import Net
import torch.nn.functional as F
import cv2
from torchvision.utils import save_image
from util import getClasses
import argparse

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w))
        cam = np.matmul(weight_softmax[idx], beforeDot)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def visualizeExample(model_path='models/training4.pth', output_name='img1'):
    # UnNormalize function to restore original image
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    model = getModel()
    mod = getMod()

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    params = list(Net().parameters())
    weight_softmax = np.squeeze(params[-1].data.numpy())

    test_loader = getTestLoader()
    test_iter = iter(test_loader)
    images, labels = test_iter.next()

    logit = model(images)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.detach().numpy()
    idx = idx.numpy()
    print("Prediction Probabilities: "+str(probs))
    print("Predicted Indices       : "+str(idx))
    predicted_idx = idx[0] # Index of predicted label
    real_idx = labels.item() # Index of real label

    classes = getClasses()
    print("Predicted: "+str(classes[predicted_idx])+" ("+str(predicted_idx)+")")
    print("Real     : "+str(classes[real_idx])+" ("+str(real_idx)+")")

    logitModel = logit.cpu().detach().numpy()

    features_blobs = mod(images)
    features_blobs1 = features_blobs.cpu().detach().numpy()
    features_blobs1_avgpool = features_blobs.view(512,7*7).mean(1).view(1,-1)
    features_blobs1_avgpool = features_blobs1_avgpool.cpu().detach().numpy()
    logitManual = np.matmul(features_blobs1_avgpool, weight_softmax.transpose())
    CAMs = returnCAM(features_blobs1, weight_softmax, [idx[0]])

    unnorm_image = unorm(images) # denormalized image
    save_image(unnorm_image, output_name+'.png')
    img = cv2.imread(output_name+'.png')
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.5 + img * 0.5
    saveImg = output_name+'_cam.png'
    cv2.imwrite(saveImg, result)

# MAIN part: parsing arguments and call visualization function
parser = argparse.ArgumentParser()
parser.add_argument('-modelpath', action='store', dest='model_path',
                    help='Path of the trained model')
parser.add_argument('-output', action='store', dest='output_name',
                    help='Name of the output image')
results = parser.parse_args()

if (results.model_path and results.output_name):
    visualizeExample(results.model_path, results.output_name)
elif (results.model_path):
    visualizeExample(model_path=results.model_path)
elif (results.output_name):
    visualizeExample(output_name=results.output_name)
else:
    visualizeExample()
