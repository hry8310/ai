# coding='utf-8'
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil
import cv2
import random
import colorsys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import torch
import torch.nn as nn

from lib.config import config

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from lib.model import Yolo3 
from lib.loss import Loss
from lib.utils import max_supp, bbox_iou , draw_box



def test(config):
    is_training = False
    net = Yolo3(config, is_training=is_training)
    net.train(is_training)

    net = nn.DataParallel(net)
    if config["test_weight"]:
        logging.info("load checkpoint from {}".format(config["test_weight"]))
        state_dict = torch.load(config["test_weight"],map_location='cpu')
        net.load_state_dict(state_dict)
    else:
        raise Exception("missing test weight!!!")

    yolo_losses = []
    olo_losses=Loss(config)

    images = []
    images_origin = []
    _image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if _image is None:
        logging.error("read path error: {}. skip it.".format(img_path))
        return 
    image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
    images_origin.append(image)  # keep for save result
    image = cv2.resize(image, (config["img_w"], config["img_h"]),
                         interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    images.append(image)
    images = np.asarray(images)
    images = torch.from_numpy(images)
    with torch.no_grad():
        outputs = net(images)
    print(outputs)

    logging.info("Save  to ./output/")    

img_path='./images/test4.jpg'
def main():
    # Start test
    test(config)


if __name__ == "__main__":
    main()
