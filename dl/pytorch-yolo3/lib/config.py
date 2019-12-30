config = \
{
    "lr": {
        "backbone_lr": 0.001,
        "other_lr": 0.01,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.1,
        "decay_step": 20,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-05,
    },
    "anchors": [[[116, 90], [156, 198], [373, 326]],
               [[30, 61], [62, 45], [59, 119]],
               [[10, 13], [16, 30], [33, 23]]],
    "classes": 20,
    "batch_size": 2,
    "data_path": "./data/train_annotation.txt",
    "pretrained_weight": "./weights/darknet53_weights_pytorch.pth", #  set empty to disable
    "darknet_type": "darknet_53",
    "epochs": 100000,
    "img_h": 416,
    "img_w": 416,
    "working_dir": "logs",              #  replace with your working dir
    "images_path": "./images/",
    "classes_names_path": "./data/voc.names",
    "test_weight": "./weights/model.pth",
    "conf_thres":0.5,
}
