import torch
import torch.nn as nn
from collections import OrderedDict

from .backbone import backbone_fn


class Yolo3(nn.Module):
    def __init__(self, config, is_training=True):
        super(Yolo3, self).__init__()
        self.config = config
        self.training = is_training
        #  backbone
        _backbone_fn = backbone_fn[self.config["darknet_type"]]
        #self.backbone = _backbone_fn(self.config["pretrained_weight"])
        self.backbone = _backbone_fn('')
        bone_outs = self.backbone.layers_out_filters
        

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

   

