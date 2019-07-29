#!/usr/bin/env python
# coding: utf-8

import ConvNet
import re
import argparse
import glob
import os
import torch.nn as nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--path',help='path to train directory',type=str,default='/exports/eddie/scratch/s1217815/AerialImageDataset/train/')
parser.add_argument('--model_path', help='path to saved models',type=str, default='/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/saved_models/')
parser.add_argument('--out_dir',help='path to results directory',type=str,default='/home/s1217815') 

def model_accuracy(models,image_paths, target_paths, results_dir):
    accList = []
    for model in models:
        print ('Path to model {}'.format(model))
        base_model = os.path.basename(model)
        batch_size = get_batch(base_model)
        train_loader, valid_loader, test_loader = ConvNet.train_valid_test_split(image_paths, target_paths,batch_size)
        print('Model: {}'.format(base_model))
        match = re.search('arch(\d+)', base_model)
        net_size = match.group(1)
        net_size = int(net_size)
        lr = re.search('lr(\d+\.\d+)', base_model)
        lr = float(lr.group(1))
        batch = get_batch(base_model)
        net = ConvNet.Net(net_size)
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(model,map_location=lambda storage, loc: storage))
        #net.cuda()
        accuracy = ConvNet.model_eval(test_loader, net,batch, lr, net_size, results_dir)
        accList.append(accuracy)
    maxInd = accList.index(max(accList))
    print('Highest accuracy: {} for model: {}'.format(accList[maxInd],models[maxInd]))
    models[maxInd]
    return maxInd

def get_batch(model):
    match = re.search('batch(\d+)',model)
    batch_size = match.group(1)
    batch_size = int(batch_size)
    return batch_size

if __name__ == "__main__":
    args=parser.parse_args()
    results_dir = args.out_dir
    gd_models = glob.glob(args.model_path+'/*.pt')
    image_paths, target_paths = glob.glob(args.path+'images/*.tif'), glob.glob(args.path+'gt/*.tif')
    maxInd = model_accuracy(gd_models, image_paths, target_paths, results_dir)    
    

