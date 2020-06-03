import torch
import torch.nn as nn
from collections import OrderedDict
from .net import Net 


class ImgCls(nn.Module):
    def __init__(self, config, is_training=True):
        super(ImgCls, self).__init__()
        self.config = config
        self.training = is_training
        self.backbone = Net()
        m_layer_out0 = 2 
        self.m_our_conv = self._m_our_conv(1024,1, m_layer_out0)

        
    def _m_our_conv(self,size, kn,out):
        return nn.Sequential(
            nn.Linear(256*52*52, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
            nn.BatchNorm1d(2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x0 = self.backbone(x)
        m_out = self.m_our_conv(x0.view(-1,256*52*52))
        return m_out        

   

