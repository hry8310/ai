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
from lib.model import Yolo3 
from lib.loss import Loss
from lib.voc import VOCDataset
from lib.config import config

def train(config):
    config["global_step"] = config.get("start_step", 0)
    is_training = True
    print(config["epochs"])

    yolo = Yolo3(config, is_training=is_training)
    yolo.train(is_training)

    # Optimizer and learning rate
    optimizer = conf_op(config, yolo)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr"]["decay_step"],
        gamma=config["lr"]["decay_gamma"])

    # Set data parallel
    yolo = nn.DataParallel(yolo)



    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(Loss(config,i)),
    # DataLoader
    dataloader = torch.utils.data.DataLoader(VOCDataset(config["data_path"],
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
            outputs = yolo(images)
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], labels)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]
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
                for i, name in enumerate(losses_name):
                    value = _loss if i == 0 else losses[i]
                    config["tensorboard_writer"].add_scalar(name,
                                                            value,
                                                            config["global_step"])

            if step > 0 and step % 100 == 0:
                # net.train(False)
                _save_checkpoint(yolo.state_dict(), config)
                # net.train(True)

        lr_scheduler.step()

    # net.train(False)
    _save_checkpoint(yolo.state_dict(), config)
    # net.train(True)
    logging.info("Bye~")

# best_eval_result = 0.0
def _save_checkpoint(state_dict, config, evaluate_func=None):
    checkpoint_path = os.path.join(config["sub_working_dir"], "model.pth")
    torch.save(state_dict, checkpoint_path)
    logging.info("Model checkpoint saved to %s" % checkpoint_path)


def conf_op(config, yolo):
    optimizer = None

    # Assign different lr for each layer
    params = None
    base_params = list(
        map(id, yolo.backbone.parameters())
    )
    logits_params = filter(lambda p: id(p) not in base_params, yolo.parameters())

    if not config["lr"]["freeze_backbone"]:
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
            {"params": yolo.backbone.parameters(), "lr": config["lr"]["backbone_lr"]},
        ]
    else:
        logging.info("freeze backbone's parameters.")
        for p in yolo.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
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

    # Creat tf_summary writer
    config["tensorboard_writer"] = SummaryWriter(sub_working_dir)
    logging.info("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir))

    # Start training
    train(config)

if __name__ == "__main__":
    main()
