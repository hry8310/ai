import torch
import torch.nn as nn
import numpy as np
import math

from lib.utils import bbox_iou , bbox_xy_iou


class Loss(nn.Module):
    def __init__(self, config,idx):
        super(Loss, self).__init__()
        self.anchors = config['anchors'][idx]
        self.num_anchors = len(self.anchors)
        self.num_classes = config['classes']
        self.bbox_attrs = 5 + self.num_classes
        self.img_size = (config["img_w"], config["img_h"]) 

        self.ignore_threshold = 0.5
        self.lambda_iou = 0.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        prediction = input.view(bs,  self.num_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(prediction[..., 0])          
        y = torch.sigmoid(prediction[..., 1])          
        w = prediction[..., 2]                         
        h = prediction[..., 3]                         
        conf = torch.sigmoid(prediction[..., 4])       
        pred_cls = torch.sigmoid(prediction[..., 5:]) 


        if targets is not None:
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            mask, noobj_mask, tx, ty, tw, th, kx,ky,kw,kh,tconf, tcls = self._decode(targets, scaled_anchors,
                                                                           in_w, in_h,
                                                                           self.ignore_threshold)

            grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
                bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(
                bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
            
            hx= x.data + grid_x
            hy = y.data + grid_y
            hw = torch.exp(w.data) * anchor_w
            hh = torch.exp(h.data) * anchor_h
            pred_boxes = FloatTensor(prediction[..., :4].shape)

     
            #loss_iou=bbox_xy_iou(hx*mask ,hy*mask, hw*mask,hh*mask,kx*mask,ky*mask, kw*mask,kh*mask)
            loss_iou=bbox_xy_iou(hx ,hy, hw,hh,kx,ky, kw,kh)
            loss_iou=1-loss_iou
            loss_iou=loss_iou*mask
            loss_iou=torch.sum(loss_iou)            

            loss_conf = self.bce_loss(conf * mask, mask) + \
                0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
            loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])
  

            loss = loss_iou * self.lambda_iou + \
                loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

            return loss,  loss_iou.item(),\
                loss_conf.item(), loss_cls.item()
        else:
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
                bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(
                bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

    def _decode(self, target, anchors, in_w, in_h, ignore_threshold):
        bs = target.size(0)

        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        kx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ky = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        kw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        kh = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)
        for b in range(bs):
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    continue
                gx = target[b, t, 1] * in_w
                gy = target[b, t, 2] * in_h
                gw = target[b, t, 3] * in_w
                gh = target[b, t, 4] * in_h
                gi = int(gx)
                gj = int(gy)
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0
                best_n = np.argmax(anch_ious)

                mask[b, best_n, gj, gi] = 1
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
                kx[b, best_n, gj, gi] = gx
                ky[b, best_n, gj, gi] = gy 
                kw[b, best_n, gj, gi] = gw
                kh[b, best_n, gj, gi] = gh
                tconf[b, best_n, gj, gi] = 1
                tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1

        return mask, noobj_mask, tx, ty, tw, th,kx,ky,kw,kh, tconf, tcls
