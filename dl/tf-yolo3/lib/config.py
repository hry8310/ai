class Config(object):
    Classes                = "./data/classes/coco.names"

    Anchors                = "./data/anchors/basline_anchors.txt"
    Avg_decay       = 0.9995
    Strides                = [8, 16, 32]
    Anchor_scale       = 3
    Iou_thre        = 0.5
    Upsample_method        = "resize"
 
    Tr_obj_scale  =5
    Tr_noobj_scale=1
    Tr_img_path            = "./data/dataset/voc_train.txt"
    Tr_batch_size            = 6
    Tr_input_size            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
    Lr_begin       = 1e-4
    Lr_end        = 1e-6
    Warm_ep         = 2
    Init_ep     = 10
    First_ep= 20
    Second_ep= 30
    Init_weight= "./logs/yolov3_coco_demo.ckpt"
    Save_weight_dir='./logs/'
    


    Te_img_path             = "./data/dataset/voc_test.txt"
    Te_batch_size             = 6 
    Te_input_size             = 544
    Te_weight_file            = "./logs/yolov3_loss=9451.8496.ckpt-1"
    Te_score_thre= 0.3
    Te_iou_thre= 0.45


cfg=Config()



