
import os
import cv2
import random
import numpy as np
import tensorflow as tf
import lib.utils as utils
from lib.config import cfg



class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.img_path  = cfg.Tr_img_path if dataset_type == 'train' else cfg.Te_img_path
        #self.input_size = cfg.Tr_input_size if dataset_type == 'train' else cfg.Te_input_size
        self.batch_size  = cfg.Tr_batch_size if dataset_type == 'train' else cfg.Te_batch_size

        self.train_input_sizes = cfg.Tr_input_size
        self.strides = np.array(cfg.Strides)
        self.classes = utils.read_class_names(cfg.Classes)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.Anchors))
        self.anchor_per_scale = cfg.Anchor_scale
        self.max_box_per_scale = 150

        self.img_box = self.load_imgs(dataset_type)
        self.num_samples = len(self.img_box)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0


    def load_imgs(self, dataset_type):
        with open(self.img_path, 'r') as f:
            txt = f.readlines()
            img_box = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(img_box)
        return img_box 

    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)
            self.train_output_sizes = self.train_input_size // self.strides

            images = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))

            label_sbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            label_mbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes))
            label_lbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes))

            sboxes = np.zeros((self.batch_size, self.max_box_per_scale, 4))
            mboxes = np.zeros((self.batch_size, self.max_box_per_scale, 4))
            lboxes = np.zeros((self.batch_size, self.max_box_per_scale, 4))

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    img_box = self.img_box[index]
                    _image, _boxes = self.parse_img(img_box)
                    _label_sbox, _label_mbox, _label_lbox, _sboxes, _mboxes, _lboxes = self.preprocess_boxes(_boxes)

                    images[num, :, :, :] = _image
                    label_sbox[num, :, :, :, :] = _label_sbox
                    label_mbox[num, :, :, :, :] = _label_mbox
                    label_lbox[num, :, :, :, :] = _label_lbox
                    sboxes[num, :, :] = _sboxes
                    mboxes[num, :, :] = _mboxes
                    lboxes[num, :, :] = _lboxes
                    num += 1
                self.batch_count += 1
                return images, label_sbox, label_mbox, label_lbox, sboxes, mboxes, lboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.img_box)
                raise StopIteration


    def parse_img(self, img_box):

        line = img_box.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = np.array(cv2.imread(image_path))
        boxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

        image, boxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(boxes))
        return image, boxes


    def box_iou(self, x1, x2):
        
        x1 = np.array(x1)
        x2 = np.array(x2)

        x1 = np.concatenate([x1[..., :2] - x1[..., 2:] * 0.5,
                                x1[..., :2] + x1[..., 2:] * 0.5], axis=-1)
        x2 = np.concatenate([x2[..., :2] - x2[..., 2:] * 0.5,
                                x2[..., :2] + x2[..., 2:] * 0.5], axis=-1)
        return utils.box_iou(x1,x2)

    def preprocess_boxes(self, boxes):

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        boxes_xywh = [np.zeros((self.max_box_per_scale, 4)) for _ in range(3)]
        box_count = np.zeros((3,))

        for box in boxes:
            box_coor = box[:4]
            box_class_ind = box[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[box_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            box_xywh = np.concatenate([(box_coor[2:] + box_coor[:2]) * 0.5, box_coor[2:] - box_coor[:2]], axis=-1)
            box_xywh_scaled = 1.0 * box_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(box_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.box_iou(box_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(box_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = box_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    box_ind = int(box_count[i] % self.max_box_per_scale)
                    boxes_xywh[i][box_ind, :4] = box_xywh
                    box_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                _anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                _detect = int(_anchor_ind / self.anchor_per_scale)
                _anchor = int(_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(box_xywh_scaled[_detect, 0:2]).astype(np.int32)

                label[_detect][yind, xind, _anchor, :] = 0
                label[_detect][yind, xind, _anchor, 0:4] = box_xywh
                label[_detect][yind, xind, _anchor, 4:5] = 1.0
                label[_detect][yind, xind, _anchor, 5:] = smooth_onehot

                box_ind = int(box_count[_detect] % self.max_box_per_scale)
                boxes_xywh[_detect][box_ind, :4] = box_xywh
                box_count[_detect] += 1
        label_sbox, label_mbox, label_lbox = label
        sboxes, mboxes, lboxes = boxes_xywh
        return label_sbox, label_mbox, label_lbox, sboxes, mboxes, lboxes

    def __len__(self):
        return self.num_batchs




