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
        self.backbone = _backbone_fn(self.config["pretrained_weight"])
        bone_outs = self.backbone.layers_out_filters
        
        # view 0
        yolo_layer_out0 = len(config["anchors"][0]) * (5 + config["classes"])
        self.yolo_convs0 = self._convs([512, 1024],bone_outs[-1], yolo_layer_out0)
        # view1
        yolo_layer_out1 = len(config["anchors"][1]) * (5 + config["classes"])
        self.yolo_conv1 = self._conv(512, 256, 1)
        self.yolo_up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.yolo_convs1 = self._convs([256, 512], bone_outs[-2] + 256, yolo_layer_out1)
        # view2
        yolo_layer_out2 = len(config["anchors"][2]) * (5 + config["classes"])
        self.yolo_conv2= self._conv(256, 128, 1)
        self.yolo_up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.yolo_convs2 = self._convs([128, 256], bone_outs[-3] + 128, yolo_layer_out2)
        

    def _conv(self, _in, _out, ks):
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _convs(self, sizes, kn, out):
        m = nn.ModuleList([
            self._conv(kn, sizes[0], 1),
            self._conv(sizes[0], sizes[1], 3),
            self._conv(sizes[1], sizes[0], 1),
            self._conv(sizes[0], sizes[1], 3),
            self._conv(sizes[1], sizes[0], 1),
            self._conv(sizes[0], sizes[1], 3)])
        m.add_module("conv_out", nn.Conv2d(sizes[1], out, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m

    def _to_view(self,layers,input):
        for i, conv in enumerate(layers):
            input = conv(input)
            if i == 4:
                output =input 
        return input, output
        

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        out0, output0 = self._to_view(self.yolo_convs0, x0)
        x1_in = self.yolo_conv1(output0)
        x1_in = self.yolo_up1(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, output1 = self._to_view(self.yolo_convs1, x1_in)
        x2_in = self.yolo_conv2(output1)
        x2_in = self.yolo_up2(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, output2 = self._to_view(self.yolo_convs2, x2_in)
        return out0, out1, out2
   

