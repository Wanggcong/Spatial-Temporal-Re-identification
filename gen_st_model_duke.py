# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision import datasets
import os
import scipy.io
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_dir',default="/home/sdb1/huangpg/st-reid/st_baseline/Duke/pytorch/",type=str, help='./train_data')
parser.add_argument('--name', default='ft_ResNet50_duke_pcb', type=str, help='save model path')

opt = parser.parse_args()
name = opt.name
data_dir = opt.data_dir


def get_id(img_path):
    camera_id = []
    labels = []
    frames = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        frame = filename[9:16]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
        frames.append(int(frame))
    return camera_id, labels, frames

def spatial_temporal_distribution(camera_id, labels, frames):
    spatial_temporal_sum = np.zeros((702,8))                       
    spatial_temporal_count = np.zeros((702,8))
    eps = 0.0000001
    interval = 100.0
    
    for i in range(len(camera_id)):
        label_k = labels[i]                 #### not in order, done
        cam_k = camera_id[i]-1              ##### ### ### ### ### ### ### ### ### ### ### ### # from 1, not 0
        frame_k = frames[i]
        spatial_temporal_sum[label_k][cam_k]=spatial_temporal_sum[label_k][cam_k]+frame_k
        spatial_temporal_count[label_k][cam_k] = spatial_temporal_count[label_k][cam_k] + 1
    spatial_temporal_avg = spatial_temporal_sum/(spatial_temporal_count+eps)          # spatial_temporal_avg: 702 ids, 8cameras, center point
    
    distribution = np.zeros((8,8,3000))
    for i in range(702):
        for j in range(8-1):
            for k in range(j+1,8):
                ###################################################### added
                if spatial_temporal_count[i][j]==0 or spatial_temporal_count[i][k]==0:
                    continue                   
                st_ij = spatial_temporal_avg[i][j]
                st_ik = spatial_temporal_avg[i][k]
                if st_ij>st_ik:
                    diff = st_ij-st_ik
                    hist_ = int(diff/interval)
                    distribution[j][k][hist_] = distribution[j][k][hist_]+1     # [big][small]
                else:
                    diff = st_ik-st_ij
                    hist_ = int(diff/interval)
                    distribution[k][j][hist_] = distribution[k][j][hist_]+1
    
    sum_ = np.sum(distribution,axis=2)
    for i in range(8):
        for j in range(8):
            distribution[i][j][:]=distribution[i][j][:]/(sum_[i][j]+eps)
    
    return distribution                    # [to][from], to xxx camera, from xxx camera

transform_train_list = [
        transforms.Resize(144, interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]


image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,transform_train_list) for x in ['train_all']}
train_path = image_datasets['train_all'].imgs
train_cam, train_label, train_frames = get_id(train_path)

train_label_order = []
for i in range(len(train_path)):
    train_label_order.append(train_path[i][1]) 


# distribution = spatial_temporal_distribution(train_cam, train_label, train_frames)
distribution = spatial_temporal_distribution(train_cam, train_label_order, train_frames)
result = {'distribution':distribution}
scipy.io.savemat('model/'+name+'/'+'pytorch_result2.mat',result)