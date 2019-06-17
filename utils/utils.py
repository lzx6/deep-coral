# -*- coding: utf-8 -*-
"""
@ project: Deep_Coral
@ author: lzx
@ file: utils.py
@ time: 2019/6/16 10:28
"""
# -*- coding: utf-8 -*-

import torch
from torch.utils import data
from torchvision import datasets, transforms
import pickle

# office path
Amazon_ImagePath = 'F:/刘子绪/数据/数据image/office31/office31/amazon/images'
Amazon_procee_path = 'F:/刘子绪/数据/数据image/office31/office31/amazon/process'
dlsr_ImagePath = 'F:/刘子绪/数据/数据image/office31/office31/dslr/images'
dlsr_procee_path = 'F:/刘子绪/数据/数据image/office31/office31/dslr/process'
webcam_ImagePath = 'F:/刘子绪/数据/数据image/office31/office31/webcam/images'
webcam_procee_path = 'F:/刘子绪/数据/数据image/office31/office31/webcam/process'

def get_data_mean_and_std(path):
    dataset = datasets.ImageFolder(path,
                                   transform=transforms.ToTensor())
    dataloader = data.DataLoader(dataset,batch_size=1)
    mean = [0,0,0]
    std = [0,0,0]
    print(len(dataset))
    for i in range(3):
        mean_every = 0
        std_every = 0
        for _,(xs,_) in enumerate(dataloader):
            img = xs[0][i].numpy()
            mean_every += img.mean()
            std_every += img.std()
        mean[i] = mean_every/len(dataset)
        std[i] = std_every/len(dataset)
    return mean,std

# x,y = get_data_mean_and_std(dlsr_ImagePath)
# print(x,y)
def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print('[INFO] Object saved to {}'.format(path))