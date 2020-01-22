import torch
import torch.nn as nn
import numpy as np
import math

from lib.utils import bbox_iou , bbox_xy_iou


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()

        self.bce_loss = nn.BCELoss()

    def forward(self, input, targets=None):
               
        if targets is not None:
            #_input= torch.sigmoid(input)
            #input= nn.log_softmax(input)
            #print('xxxxx-chag')
            #print(_input)
            loss_cls = self.bce_loss(input, targets)
            return loss_cls
        else:
            return input

