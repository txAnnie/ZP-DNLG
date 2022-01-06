# !usr/bin/python3
# -*- coding:utf-8 -*-
# Author: xtan
# File: Discriminator.py
# Time: 2020/4/6 9:33
import torch.nn as nn
import numpy as np
import torch
import onmt.opts
from onmt.modules.GAN_cnn import GAN_cnn
from config import *


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.cnd = GAN_cnn()
        down_dim = 256
        self.fc_3 = nn.Sequential(
            nn.BatchNorm1d(down_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Linear(down_dim, down_dim // 2),
            nn.BatchNorm1d(down_dim // 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Linear(down_dim // 2, 1),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(down_dim, down_dim // 2),
            nn.BatchNorm1d(down_dim // 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Linear(down_dim // 2, 1),
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(down_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Linear(down_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
 
        if dis_layer == 'UNI':
            out = self.fc(data)
        if dis_layer == 'BI':
            out = self.fc_2(data)
        if dis_layer == 'MBI':
            out = self.fc_3(data)
        return out
