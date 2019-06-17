# -*- coding: utf-8 -*-
"""
@ project: Deep_Coral
@ author: lzx
@ file: dataloader.py
@ time: 2019/6/16 10:53
"""
import torch
from torch.utils import data
from torchvision import datasets, transforms

# office path
Amazon_ImagePath = 'F:/数据/数据image/office31/office31/amazon/images'
Amazon_procee_path = 'F:/数据/数据image/office31/office31/amazon/process'
dlsr_ImagePath = 'F:/数据/数据image/office31/office31/dslr/images'
dlsr_procee_path = 'F:/数据/数据image/office31/office31/dslr/process'
webcam_ImagePath = 'F:/数据/数据image/office31/office31/webcam/images'
webcam_procee_path = 'F:/数据/数据image/office31/office31/webcam/process'

def office31_loader(name,batch_size):
    print('now load {} dataset......'.format(name))
    datas_path = {'Amazon':Amazon_ImagePath,
                  'Dlsr':dlsr_ImagePath,
                  'Webcam':webcam_ImagePath}
    means = {'Amazon':[0.7923507540783308, 0.7862063347129564, 0.7841796530691664],
            'Dlsr':[0.4708635708294719, 0.44865584816319876, 0.4063774835034068],
             'Webcam':[0.6119798301150964, 0.6187647400037297, 0.6172966210347302],
             'imagenet':[0.485,0.456,0.406]}
    stds = {'Amazon':[0.2769164364331362, 0.28152348841965347, 0.2828729676283079],
            'Dlsr':[0.18538173842412162, 0.17889121580255557, 0.18190332775539064],
             'Webcam':[0.22763857108616978, 0.23339382150450594, 0.23722725519031848],
            'imagenet':[0.229,0.224,0.406]}
    imgsize_W = 224
    imgsize_H = 224
    transform = [transforms.Resize((imgsize_W,imgsize_H)),
                 transforms.ToTensor(),
                 transforms.Normalize(means[name],stds[name])]
    dataset = datasets.ImageFolder(datas_path[name],transform = transforms.Compose(transform))
    # print(len(dataset))
    dataloader = data.DataLoader(
        dataset= dataset,
        batch_size=batch_size,
        shuffle = True,
        drop_last= True
    )
    return dataloader
# office31_loader('Webcam',128)