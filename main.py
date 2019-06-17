# -*- coding: utf-8 -*-
"""
@ project: Deep_Coral
@ author: lzx
@ file: main.py
@ time: 2019/6/16 15:16
"""
from data import office31_loader
from params import param
from models import CORAL,Deep_coral,AlexNet
from utils import save
import torch
import torch.nn as nn
import torchvision.models.alexnet as ALEXNET
args = param()

src_loader = office31_loader('Amazon',batch_size=args.train_batch)
tgt_loader = office31_loader('Webcam',batch_size=args.test_batch)
criterion = nn.CrossEntropyLoss()
# print(len(src_loader),len(tgt_loader))

def train(model,optimizer,epoch,lambda_):
    result = []
    # src,tgt = list(enumerate(src_loader)),list(enumerate(tgt_loader))
    train_steps = min(len(src_loader),len(tgt_loader))
    # print(train_steps)?
    iter_target = iter(tgt_loader)
    iter_source = iter(src_loader)
    for i in range(train_steps):
        # _,(src_data,src_label) = src[i]
        src_data,src_label = iter_source.next()
        if i % len(tgt_loader) == 0:
            iter_target = iter(tgt_loader)
        tgt_data, _ = iter_target.next()
        # _,(tgt_data,_) = tgt[int(i%len(tgt))]
        if torch.cuda.is_available():
            src_data = src_data.cuda()
            tgt_data = tgt_data.cuda()
            src_label = src_label.cuda()

        optimizer.zero_grad()
        src_out,tgt_out = model(src_data,tgt_data)

        loss_classifier = criterion(src_out,src_label)
        loss_coral = CORAL(src_out,tgt_out)

        sum_loss = lambda_*loss_coral+loss_classifier
        sum_loss.backward()
        optimizer.step()

        result.append({
            'epoch': epoch,
            'step': i + 1,
            'total_steps': train_steps,
            'lambda': lambda_,
            'coral_loss': loss_coral.item(),
            'classification_loss': loss_classifier.item(),
            'total_loss': sum_loss.item()
        })
        print('Train Epoch: {:2d} [{:2d}/{:2d}]\t'
              'Lambda: {:.4f}, Class: {:.6f}, CORAL: {:.6f}, Total_Loss: {:.6f}'.format(
            epoch,
            i + 1,
            train_steps,
            lambda_,
            loss_classifier.item(),
            loss_coral.item(),
            sum_loss.item()
        ))
    return result


def test(model,dataset_loader,every_epoch):
    model.eval()
    test_loss = 0
    corrcet = 0
    for tgt_data,tgt_label in dataset_loader:
        if torch.cuda.is_available():
            tgt_data = tgt_data.cuda()
            tgt_label = tgt_label.cuda()

        tgt_out,_ = model(tgt_data,tgt_data)
        test_loss = criterion(tgt_out,tgt_label).item()
        pred = tgt_out.data.max(1,keepdim=True)[1]
        corrcet += pred.eq(tgt_label.data.view_as(pred)).cpu().sum()
    test_loss /= len(dataset_loader)
    return {
        'epoch': every_epoch,
        'average_loss': test_loss,
        'correct': corrcet,
        'total': len(dataset_loader.dataset),
        'accuracy': 100. * float(corrcet) / len(dataset_loader.dataset)
    }

def load_pretrained(model):

    alexnet = ALEXNET(pretrained= True).state_dict()# 下载预训练模型
    model_dict = model.state_dict()#本身模型的参数

    pretrained_dict = {k:v for k,v in alexnet.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    model = Deep_coral(num_classes=31)
    optimizer = torch.optim.SGD([{'params': model.feature.parameters()},
                                 {'params':model.fc.parameters(),'lr':10*args.lr}],
                                lr= args.lr,momentum=args.momentum,weight_decay=args.weight_clay)
    if torch.cuda.is_available():
        model = model.cuda()
    load_pretrained(model.feature)
    training_sta = []
    test_s_sta = []
    test_t_sta = []
    for e in range(args.epochs):
        # lambda_ = (e+1)/args.epochs
        lambda_ = 0.0
        res = train(model,optimizer,e+1,lambda_)
        print('###EPOCH {}: Class: {:.6f}, CORAL: {:.6f}, Total_Loss: {:.6f}'.format(
            e + 1,
            sum(row['classification_loss'] / row['total_steps'] for row in res),
            sum(row['coral_loss'] / row['total_steps'] for row in res),
            sum(row['total_loss'] / row['total_steps'] for row in res),
        ))
        training_sta.append(res)
        test_source = test(model, src_loader, e)
        test_target = test(model, tgt_loader, e)
        test_s_sta.append(test_source)
        test_t_sta.append(test_target)
        print('###Test Source: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            e + 1,
            test_source['average_loss'],
            test_source['correct'],
            test_source['total'],
            test_source['accuracy'],
        ))
        print('###Test Target: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            e + 1,
            test_target['average_loss'],
            test_target['correct'],
            test_target['total'],
            test_target['accuracy'],
        ))
    result_path = 'result_norm_no'
    import os
    os.makedirs(result_path,exist_ok=True)
    torch.save(model.state_dict(),result_path+'/checkpoint.tar')
    save(training_sta,result_path+'/training_state.pkl')
    save(test_s_sta, result_path + '/test_s_sta.pkl')
    save(test_t_sta, result_path + '/test_t_sta.pkl')






