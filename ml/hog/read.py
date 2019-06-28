import tensorflow as tf
import  numpy as np
import os
import cv2
#import cv2.cv as cv


def get_files(file_dir='./dataset/train/',folders=['pos','neg']):
    imgs = []
    labels = []
    for folder in folders:
        _dir = os.path.join(file_dir,folder); 
        for file in os.listdir(_dir):
            src = cv2.imread(os.path.join(_dir,file),5)
            imgs.append(  src)
            if folder== 'pos':
                labels.append(1)
            else:
                labels.append(-1)
    return imgs,labels
