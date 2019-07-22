#!/usr/bin/env python
# coding: utf-8

import torch
import itertools as it
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
from torch.autograd import Variable
import matplotlib.patches as mpatches
import argparse
import os, gdal, osr, ogr
import re
import ConvNet


parser = argparse.ArgumentParser(description='Predict the class of each pixel for an image and save the result. Images taken from train folder and include mask')
parser.add_argument('-model',type=str,default='model_inria.pt',help='A saved pytorch model')
parser.add_argument('-inpfile',type=str,default='/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/train/images/kitsap31.tif',help='Path and filename of image to be classified')      

def raster2array(rasterfn):
    raster=gdal.Open(rasterfn)
    return raster.ReadAsArray()

def array2raster(rasterfn,newRasterfn,array):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def image_loader(rasterfn):
    """load image, returns predicted array"""
    image = raster2array(rasterfn)    
    image=np.transpose(image,(1,2,0))
    image = toTensor(image)   
    image = Variable(image, requires_grad=True) 
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.unsqueeze(0)
    output = net(image)
    variable = Variable(output) 
    if torch.cuda.is_available(): 
        variable = variable.cuda() 
    num = variable.data[0]
    num = num.permute(1,2,0)
    num = num.cpu()
    b = num.numpy()
    final_prediction=b[:,:,0]
    labels = (final_prediction > 0.5).astype(np.int)
    return labels
    
 
if __name__ == '__main__':
    args=parser.parse_args()
    
    rasterfn = args.inpfile
    base = os.path.basename(rasterfn) 
    fname = os.path.splitext(base)[0]
    newRasterfn='/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/predict_raster/predict_{}.tif'.format(fname)
    base_model = os.path.basename(args.model)
    match = re.search('arch(\d+)',base_model)
    net_size = match.group(1)
    net_size = int(net_size)
    net = ConvNet.Net(net_size)
    net = nn.DataParallel(net)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(args.model))
    else:
        net.load_state_dict(torch.load(args.model,map_location=lambda storage, loc: storage)) 
        
    toTensor = transforms.ToTensor()
    array = image_loader(rasterfn)
    array2raster(rasterfn,newRasterfn,array)
