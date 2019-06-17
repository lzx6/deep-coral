# -*- coding: utf-8 -*-
"""
@ project: Deep_Coral
@ author: lzx
@ file: params.py
@ time: 2019/6/16 15:16
"""

class param():
    def __init__(self):
        self.lr = 1e-3
        self.weight_clay = 5e-4
        self.momentum = 0.9
        self.train_batch = 128
        self.test_batch = 128
        self.epochs = 50
