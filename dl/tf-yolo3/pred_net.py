#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate.py
#   Author      : YunYang1994
#   Created date: 2019-02-21 15:30:26
#   Description :
#
#================================================================

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import lib.utils as utils
from lib.config import cfg
from lib.model import Yolo3

class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.Te_input_size
        self.anchor_per_scale = cfg.Anchor_scale
        self.classes          = utils.read_class_names(cfg.Classes)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.Anchors))
        self.score_threshold  = cfg.Te_score_thre
        self.iou_threshold    = cfg.Te_iou_thre
        self.moving_ave_decay = cfg.Avg_decay
        self.annotation_path  = cfg.Te_img_path
        self.weight_file      = cfg.Te_weight_file

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')

        model = Yolo3(self.input_data, self.trainable)
        self.p_sbox, self.p_mbox, self.p_lbox = model.p_sbox, model.p_mbox, model.p_lbox

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        p_sbox, p_mbox, p_lbox = self.sess.run(
            [self.p_sbox, self.p_mbox, self.p_lbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            }
        )

        p_box = np.concatenate([np.reshape(p_sbox, (-1, 5 + self.num_classes)),
                                    np.reshape(p_mbox, (-1, 5 + self.num_classes)),
                                    np.reshape(p_lbox, (-1, 5 + self.num_classes))], axis=0)
        boxes = utils.postprocess_boxes(p_box, (org_h, org_w), self.input_size, self.score_threshold)
        boxes = utils.nms(boxes, self.iou_threshold)

        return boxes

    def get_img(self,image):
        #image = cv2.imread(img_path)  
        boxes=self.predict(image)
        return boxes;
        #image = utils.draw_bbox(image, boxes)
        #cv2.imwrite(img_opath, image)




img_name='test4.jpg'
img_path='./test/'+img_name
img_opath='./otest/o_'+img_name
 
#YoloTest().get_img()



