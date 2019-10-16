import numpy as np
import tensorflow as tf
import lib.utils as utils
import lib.net as net 
from lib.config import cfg


class Yolo3(object):

    def __init__(self, trainable):

        self.trainable        = trainable
        self.classes          = utils.read_class_names(cfg.Classes)
        self.num_class        = len(self.classes)
        self.strides          = np.array(cfg.Strides)
        self.anchors          = utils.get_anchors(cfg.Anchors)
        self.anchor_scale = cfg.Anchor_scale
        self.iou_loss_thresh  = cfg.Iou_thre
        self.upsample_method  = cfg.Upsample_method
        self.input_data   = tf.placeholder(dtype=tf.float32, name='input_data')
        self.label_sbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
        self.label_mbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
        self.label_lbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
        self.true_sboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
        self.true_mboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
        self.true_lboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
        self.input_data   = tf.placeholder(dtype=tf.float32, name='input_data')

        #self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_nework(self.input_data)
        self.lbox, self.mbox, self.sbox = net.yolonet(self.input_data,self.trainable,self.num_class) 

        with tf.variable_scope('pred_sbbox'):
            self.p_sbox = self.decode(self.sbox, self.anchors[0], self.strides[0])

        with tf.variable_scope('pred_mbbox'):
            self.p_mbox = self.decode(self.mbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.p_lbox = self.decode(self.lbox, self.anchors[2], self.strides[2])



    def __build_nework_0(self, input_data):

        #route_1, route_2, input_data = net.darknet53(input_data, self.trainable)
        #conv_lbbox, conv_mbbox, conv_sbbox=net.yolonet(input_data,route_1, route_2 ,self.trainable,self.num_class)
        lbox, mbox, sbox=net.yolonet(input_data,self.trainable,self.num_class)

        return lbox, mbox, sbox



    def decode(self, output, anchors, stride):

        output_shape       = tf.shape(output)
        batch_size       = output_shape[0]
        output_size      = output_shape[1]
        anchor_scale = len(anchors)

        output = tf.reshape(output, (batch_size, output_size, output_size, anchor_scale, 5 + self.num_class))

        output = tf.reshape(output, (batch_size, output_size, output_size, anchor_scale, 5 + self.num_class))

        raw_dxdy = output[:, :, :, :, 0:2]
        raw_dwdh = output[:, :, :, :, 2:4]
        raw_conf = output[:, :, :, :, 4:5]
        raw_prob = output[:, :, :, :, 5: ]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(raw_conf)
        pred_prob = tf.sigmoid(raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)





  
    def xywh_sq(self, boxes1, boxes2):
        return tf.square(boxes1-boxes2)



    def box_iou(self,x1, x2):

        x1_area = x1[..., 2] * x1[..., 3]
        x2_area = x2[..., 2] * x2[..., 3]

        x1 = tf.concat([x1[..., :2] - x1[..., 2:] * 0.5,
                            x1[..., :2] + x1[..., 2:] * 0.5], axis=-1)
        x2 = tf.concat([x2[..., :2] - x2[..., 2:] * 0.5,
                            x2[..., :2] + x2[..., 2:] * 0.5], axis=-1)

        t = tf.maximum(x1[..., :2], x2[..., :2])
        d = tf.minimum(x1[..., 2:], x2[..., 2:])

        section = tf.maximum(d - t, 0.0)
        inter = section[..., 0] * section[..., 1]
        union = x1_area + x2_area - inter
        iou = 1.0 * inter / union

        return iou


    def loss_layer(self, output, pred, label, bboxes, anchors, stride):

        output_shape  = tf.shape(output)
        batch_size  = output_shape[0]
        output_size = output_shape[1]
        input_size  = stride * output_size
        output = tf.reshape(output, (batch_size, output_size, output_size,
                                 self.anchor_scale, 5 + self.num_class))
        raw_conf = output[:, :, :, :, 4:5]
        raw_prob = output[:, :, :, :, 5:]

        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]

        label_xywh    = label[:, :, :, :, 0:4]
        #respond_bbox  = label[:, :, :, :, 4:5]
        object_mask  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]


        input_size = tf.cast(input_size, tf.float32)

        box_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        xywh_loss = self.xywh_sq(pred_xywh, label_xywh) 

        xywh_loss = object_mask * box_loss_scale * (xywh_loss)

        iou = self.box_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        ignore_mask = (1.0 - object_mask) * tf.cast( max_iou < self.iou_loss_thresh, tf.float32 )

        conf_loss= object_mask * tf.square(pred_conf-object_mask) *cfg.Tr_obj_scale  +  ignore_mask *tf.square(pred_conf-object_mask)*cfg.Tr_noobj_scale


        prob_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=raw_prob)

        xywh_loss = tf.reduce_mean(tf.reduce_sum(xywh_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

        return xywh_loss, conf_loss, prob_loss


    def final_loss(self, label_sbox, label_mbox, label_lbox, true_sbox, true_mbox, true_lbox):

        with tf.name_scope('smaller_box_loss'):
            loss_s = self.loss_layer(self.sbox, self.p_sbox, label_sbox, true_sbox,
                                         anchors = self.anchors[0], stride = self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_m = self.loss_layer(self.mbox, self.p_mbox, label_mbox, true_mbox,
                                         anchors = self.anchors[1], stride = self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_l = self.loss_layer(self.lbox, self.p_lbox, label_lbox, true_lbox,
                                         anchors = self.anchors[2], stride = self.strides[2])

        with tf.name_scope('xywh_loss'):
            xywh_loss = loss_s[0] + loss_m[0] + loss_l[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_s[1] + loss_m[1] + loss_l[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_s[2] + loss_m[2] + loss_l[2]

        return xywh_loss, conf_loss, prob_loss





    def model_loss(self):
        self.xywh_loss,self.conf_loss,self.prob_loss=self.final_loss(
             self.label_sbox,self.label_mbox,self.label_lbox,self.true_sboxes,
             self.true_mboxes,self.true_lboxes)
        self.last_loss=self.xywh_loss+self.conf_loss+self.prob_loss
        


