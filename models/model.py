# -*- coding: utf-8 -*-
"""
@ project: Deep_Coral
@ author: lzx
@ file: model.py
@ time: 2019/6/16 11:06
"""
import torch.nn as nn
import torch

__all__ = ['CORAL','AlexNet','Deep_coral']
def CORAL(src,tgt):
    d = src.size(1)
    # xm = torch.mean(src,0,keepdim = True)-src # keepdim保持维度,按列求和
    # src_c = torch.matmul(torch.transpose(xm,0,1),xm)
    #
    # xmt = torch.mean(tgt, 0, keepdim=True) - tgt  # keepdim保持维度,按列求和
    # tgt_c = torch.matmul(torch.transpose(xmt, 0, 1), xmt) # 输出为d*d矩阵
    src_c = coral(src)
    tgt_c = coral(tgt)

    loss = torch.sum(torch.mul((src_c-tgt_c),(src_c-tgt_c)))
    loss = loss/(4*d*d)
    return loss


def coral(data):
    n = data.size(0)
    id_row = torch.ones(n).resize(1,n)
    if torch.cuda.is_available():
        id_row = id_row.cuda()
    sum_column = torch.mm(id_row,data)
    mean_column = torch.div(sum_column,n)
    mean_mean = torch.mm(mean_column.t(),mean_column)
    d_d = torch.mm(data.t(),data)
    coral_result = torch.add(d_d,(-1*mean_mean))*1.0/(n-1)
    return coral_result

class Deep_coral(nn.Module):
    def __init__(self,num_classes = 1000):
        super(Deep_coral,self).__init__()
        self.feature = AlexNet()
        self.fc = nn.Linear(4096,num_classes)
        self.fc.weight.data.normal_(0,0.005)# 原论文中设置的初始化方法

    def forward(self,src,tgt):
        src = self.feature(src)
        src = self.fc(src)
        tgt = self.feature(tgt)
        tgt = self.fc(tgt)
        return src,tgt


'''官方Alexnet网络'''
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# x = torch.randn(5,4)
# print(torch.mean(x,0))
# CORAL(x,x)