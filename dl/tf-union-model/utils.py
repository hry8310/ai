# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import cv2


def get_img(dir_path, folder_names=['cat', 'dog'], shuffle=True):
    img_path_list = []
    label_list = []
    img_count = 0
    for folder_name in folder_names:
        filenames = os.listdir(os.path.join(dir_path, folder_name))
        for f in filenames:
            img_path_list.append(os.path.join(dir_path, folder_name, f))
            label_list.append([0,1] if 'cat' == folder_name  else [1,0])
            img_count += 1
    img_path_list = np.array(img_path_list)
    label_list = np.array(label_list)
    if shuffle == True:
        index = np.random.permutation(np.arange(0, img_count, 1))
        img_path_list_shuffled = img_path_list[index]
        label_list_shuffled = label_list[index]
    else:
        img_path_list_shuffled = img_path_list
        label_list_shuffled = label_list
    return img_path_list_shuffled, label_list_shuffled

def img_resize(img_path, img_height, img_width):
    img_src = cv2.imread(img_path)
    img_resized = cv2.resize(img_src, (img_height,img_width), interpolation=cv2.INTER_CUBIC)
    return img_resized

def rgb2gray(img_rgb):
    img_gray = np.dot(img_rgb[...,:3], [0.299, 0.587, 0.114])
    img_gray = img_gray / 255.0
    return img_gray.reshape(img_rgb.shape[0], img_rgb.shape[1], 1)

def batch_iter(batch_size, num_epochs, img_path_list, label_list,
        img_height, img_width, shuffle=True):
    
    img_path_list = np.array(img_path_list)
    label_list = np.array(label_list)
    data_size = len(label_list)
    num_batches_per_epoch = int((data_size-1)/batch_size)+1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            img_path_list_shuffled = img_path_list[shuffle_indices]
            label_list_shuffled = label_list[shuffle_indices]
        else:
            img_path_list_shuffled = img_path_list
            label_list_shuffled = label_list
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num*batch_size
            end_index = min((batch_num+1)*batch_size, data_size)
            img_list_shuffled = []
            img_list_shuffled_2 = []
            for i in range(start_index, end_index):
                img_data = img_resize(img_path=img_path_list_shuffled[i], img_height=img_height, img_width=img_width)
                img_data = rgb2gray(img_data)
                img_list_shuffled.append(img_data)
                
                img_data_2 = img_resize(img_path=img_path_list_shuffled[i], img_height=img_height//2, img_width=img_width//2)
                img_data_2 = rgb2gray(img_data_2)
                img_list_shuffled_2.append(img_data_2)
            img_list_shuffled = np.array(img_list_shuffled)
            img_list_shuffled_2 = np.array(img_list_shuffled_2)
            yield img_list_shuffled, img_list_shuffled_2, label_list_shuffled[start_index:end_index]
