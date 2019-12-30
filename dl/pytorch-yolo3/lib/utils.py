from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import os 
import random
import colorsys


def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def bbox_xy_iou(cb1_x1, cb1_y1, cb1_x2, cb1_y2,cb2_x1, cb2_y1, cb2_x2, cb2_y2):
    b1_x1=cb1_x1-cb1_x2/2 
    b1_x2=cb1_x1+cb1_x2/2 
    b1_y1=cb1_y1-cb1_y2/2
    b1_y2=cb1_y1+cb1_y2/2

    b2_x1=cb2_x1-cb2_x2/2 
    b2_x2=cb2_x1+cb2_x2/2 
    b2_y1=cb2_y1-cb2_y2/2
    b2_y2=cb2_y1+cb2_y2/2
    return bbox_xy_iou_0(b1_x1, b1_y1, b1_x2, b1_y2,b2_x1, b2_y1, b2_x2, b2_y2)


def bbox_xy_iou_0(b1_x1, b1_y1, b1_x2, b1_y2,b2_x1, b2_y1, b2_x2, b2_y2):

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def max_supp(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        if not image_pred.size(0):
            continue
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            max_detections = []
            while detections_class.size(0):
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))

    return output


def draw_box(_image,classes,batch_detections,config,path='./output/test2.jpg'):
    num_classes=len(classes)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    if not os.path.isdir("./output/"):
        os.makedirs("./output/")
    for idx, detections in enumerate(batch_detections):
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # Rescale coordinates to original dimensions
                color=colors[int(cls_pred)]
                ori_h, ori_w = _image.shape[:2]
                pre_h, pre_w = config["img_h"], config["img_w"]
                box_h = ((y2 - y1) / pre_h) * ori_h
                box_w = ((x2 - x1) / pre_w) * ori_w
                y1 = (y1 / pre_h) * ori_h
                x1 = (x1 / pre_w) * ori_w
                # Create a Rectangle patch
                cv2.rectangle(_image, (x1,y1), (x1+box_w,y1+box_h), color,2)
                # Add the bbox to the plot
                # Add label
                cv2.putText(_image, classes[int(cls_pred)], (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0),  1,lineType=cv2.LINE_AA)

        cv2.imwrite(path,_image)
