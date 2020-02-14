#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import cv2
import utils
import numpy as np
import tensorflow as tf
from PIL import Image

return_elements = ["input_x:0",'dropout_keep_prob:0', "prediction/y_pred:0"]
pb_file         = "./test.pb"
image_path      = "./data/test1/test4.jpg"
#image_path      = "./docs/images/road.jpeg"
num_classes     = 80
input_size      = 416
graph           = tf.Graph()

np.set_printoptions(threshold=np.nan)
def get_one_image(img_dir):
   imgArr=[]
    
   #image = Image.open(img_dir)
    
   print(img_dir)
   #image = image.resize([224, 224])
   #image = np.array(image)
   image = utils.img_resize(img_dir,224,224)
   image = utils.rgb2gray(image)
   print(image.reshape(-1).shape)
   print(image.reshape(-1))
   #print(image[223][223])
   imgArr.append(image)
   imgArr=np.array(imgArr)
   print(imgArr.reshape(-1))
   #print(image[223][223])
   return imgArr

def read_pb(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements

image_array = get_one_image(image_path)
print(image_array.shape)

return_tensors = read_pb(graph, pb_file, return_elements)

with tf.Session(graph=graph) as sess:
    prediction = sess.run(
        [return_tensors[2]],
                feed_dict={ return_tensors[0]: image_array,return_tensors[1]:1.0})
 
    print(prediction[0])
    print('prediction  %f',prediction)
    max_index = np.argmax(prediction)

    if prediction[0]==1:
        print('This is a cat with possibility')
    else:
        print('This is a dog with possibility')



