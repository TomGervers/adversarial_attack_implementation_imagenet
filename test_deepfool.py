import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os

net = models.efficientnet_b7(pretrained=True)

# Switch to evaluation mode
net.eval()

im_orig = Image.open('12.jpeg')

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]


# Remove the mean
im = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])(im_orig)

r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)


tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                         transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                         transforms.Lambda(clip),
                         transforms.ToPILImage(),
                         transforms.CenterCrop(224)])

figure, ax = plt.subplots(1,2, figsize=(14,8))
ax[0].imshow(tf(im))
ax[0].set_title('Clean Example: {}'.format(str_label_orig[10:]), fontsize=20)

ax[1].imshow(tf(pert_image.cpu()[0]))
ax[1].set_title('Perturbed example: {}'.format(str_label_pert[10:]), fontsize=20)
plt.show()

figure, ax = plt.subplots(1,3, figsize=(14,8))
ax[0].imshow(im_orig)
ax[0].set_title('Clean Example', fontsize=20)

    
ax[2].imshow(tf(pert_image.cpu()[0]))
ax[2].set_title('Adversarial Example', fontsize=20)
    
ax[0].axis('off')
ax[2].axis('off')
    
ax[0].text(0.5,-0.13, "Prediction: {}\n Probability: {}".format(clean_pred, clean_prob), size=15, ha="center", 
         transform=ax[0].transAxes)

ax[2].text(0.5,-0.13, "Prediction: {}\n Probability: {}".format(adv_pred, adv_prob), size=15, ha="center", 
    transform=ax[2].transAxes)
    

plt.show()
