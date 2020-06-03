config = \
{
    "lr": {
        "backbone_lr": 0.00000001,
        "decay_gamma": 0.1,
        "decay_step": 20,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-08,
    },
    "batch_size": 2,
    "data_path": "./train",
    "epochs": 100000,
    "img_h": 416,
    "img_w": 416,
    "working_dir": "logs",              #  replace with your working dir
    "images_path": "./images/",
    "test_weight": "./logs/20200601173024/model.pth"
}
