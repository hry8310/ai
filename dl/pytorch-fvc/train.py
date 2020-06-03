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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
# sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'evaluate'))
from lib.model import ImgCls 
from lib.loss import Loss
from lib.dataset import ImgDataset
from lib.config import config

def train(config):
    config["global_step"] = config.get("start_step", 0)
    is_training = True
    print(config["epochs"])

    cls = ImgCls(config, is_training=is_training)
    cls.train(is_training)

    # Optimizer and learning rate
    optimizer = conf_op(config, cls)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr"]["decay_step"],
        gamma=config["lr"]["decay_gamma"])

    # Set data parallel
    cls = nn.DataParallel(cls)



    cls_losses=Loss(config)
    # DataLoader
    dataloader = torch.utils.data.DataLoader(ImgDataset(config["data_path"],
                                                         (config["img_w"], config["img_h"]),
                                                         is_training=True),
                                             batch_size=config["batch_size"],
                                             shuffle=True, num_workers=1, pin_memory=True)

    logging.info("Start training.")
    for epoch in range(config["epochs"]):
        for step, samples in enumerate(dataloader):
            images, labels = samples["image"], samples["label"]
            start_time = time.time()
            config["global_step"] += 1

            # Forward and backward
            optimizer.zero_grad()
            outputs = cls(images)
            loss = cls_losses(outputs, labels)
            loss.backward()
            optimizer.step()

            if step > 0 and step % 10 == 0:
                _loss = loss.item()
                duration = float(time.time() - start_time)
                example_per_second = config["batch_size"] / duration
                lr = optimizer.param_groups[0]['lr']
                logging.info(
                    "epoch [%.3d] iter = %d loss = %.2f example/sec = %.3f lr = %.5f "%
                    (epoch, step, _loss, example_per_second, lr)
                )
                config["tensorboard_writer"].add_scalar("lr",
                                                        lr,
                                                        config["global_step"])
                config["tensorboard_writer"].add_scalar("example/sec",
                                                        example_per_second,
                                                        config["global_step"])
                config["tensorboard_writer"].add_scalar("classloass",
                                                        _loss,
                                                        config["global_step"])

            if step > 0 and step % 100 == 0:
                # net.train(False)
                _save_checkpoint(cls.state_dict(), config)
                # net.train(True)

        lr_scheduler.step()

    # net.train(False)
    _save_checkpoint(cls.state_dict(), config)
    # net.train(True)
    logging.info("Bye~")

# best_eval_result = 0.0
def _save_checkpoint(state_dict, config, evaluate_func=None):
    checkpoint_path = os.path.join(config["sub_working_dir"], "model.pth")
    torch.save(state_dict, checkpoint_path)
    logging.info("Model checkpoint saved to %s" % checkpoint_path)


def conf_op(config, cls):
    optimizer = None

    # Assign different lr for each layer
    params = None
    base_params = list(
        map(id, cls.backbone.parameters())
    )
    logits_params = filter(lambda p: id(p) not in base_params, cls.parameters())

    params = [
         {"params": cls.parameters(), "lr": config["lr"]["backbone_lr"]},
    ]

    if config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])
    elif config["optimizer"]["type"] == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"],
                               amsgrad=True)
    elif config["optimizer"]["type"] == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])
    else:
        logging.info("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))

    return optimizer

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")


    # Create sub_working_dir
    sub_working_dir = '{}/{}'.format(
        config['working_dir'] ,
        time.strftime("%Y%m%d%H%M%S", time.localtime()))
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    config["sub_working_dir"] = sub_working_dir
    logging.info("sub working dir: %s" % sub_working_dir)

    config["tensorboard_writer"] = SummaryWriter(sub_working_dir)
    logging.info("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir))

    # Start training
    train(config)

if __name__ == "__main__":
    main()
