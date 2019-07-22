#!/usr/bin/env python
# coding: utf-8

import ConvNet
import itertools as it
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.patches as mpatches
import argparse
import os
import re
import torch

parser = argparse.ArgumentParser(description='Predict the class of each pixel for an image and save the result. Images taken from train folder and include mask')
parser.add_argument('-model',type=str,default='model_inria.pt',help='A saved pytorch model')
parser.add_argument('-inpfile',type=str,default='/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/train/images/kitsap31.tif',help='Path and filename of image to be classified')
parser.add_argument('-mask',type=str,default='/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/train/gt/kitsap31.tif',help='Path to mask in train folder')



def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    RGB_image = trans(image)
    image = Variable(image, requires_grad=True)
    if torch.cuda.is_available():                                                                     
        image = image.cuda()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image,RGB_image  #assumes that you're using GPU


def image_plotter(image,RGB_image,mask,fname):
    output = net(image)
    variable = Variable(output)
    if torch.cuda.is_available():
        variable = variable.cuda()
    num = variable.data[0]
    num = num.permute(1,2,0)
    b = num.numpy()
    final_prediction=b[:,:,0]
    labels = (final_prediction > 0.5).astype(np.int)
    fig=plt.figure(figsize=(15, 10), dpi= 300, facecolor='w', edgecolor='k')
    ax1 = plt.subplot(1,3,1)
    ax1.set_title('Original RGB_image of {}'.format(fname))
    ax2 = plt.subplot(1,3,3)
    ax2.set_title('Prediction of buildings in {}'.format(fname))
    ax3=plt.subplot(1,3,2)
    ax3.set_title('Mask of buildings in {}'.format(fname))
    im = ax2.imshow(labels, interpolation='none',cmap="binary")
    patches = [(mpatches.Patch(color='black', label="No Buildings")),(mpatches.Patch(color='white', label="Buildings"))]
    ax2.legend(handles=patches, bbox_to_anchor=(1.45, 1), loc="upper right", borderaxespad=0.,facecolor="plum" )
    ax1.imshow(RGB_image)
    ax3.imshow(mask,cmap="binary_r")
    plt.savefig('/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/predict/Predicted_{}.png'.format(fname),bbox_inches='tight',dpi=300)

if __name__ == '__main__':
    args=parser.parse_args()
    path = args.inpfile
    base = os.path.basename(path) 
    fname = os.path.splitext(base)[0]
    mask = Image.open(args.mask)
    match = re.search('arch(\d+)',args.model)
    net_size = match.group(1)
    net_size = int(net_size)
    net = ConvNet.Net(net_size)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(args.model,map_location=lambda storage, loc: storage))
    loader = transforms.Compose([ transforms.ToTensor()])
    image,RGB_image = image_loader(args.inpfile)
    image_plotter(image,RGB_image,mask,fname)
