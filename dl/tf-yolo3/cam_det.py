#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import cv2
import time
import numpy as np
import lib.utils as utils
import tensorflow as tf
from PIL import Image
import multiprocessing as mp
from lib.config import cfg
from lib.model import Yolo3

from pred_img import YoloTest
 
video_path      = "http://admin:admin@192.168.1.105:8081"
 



def read_img(q):
    vid = cv2.VideoCapture(video_path)
    while True:
        fnum=vid.get(cv2.CAP_PROP_FRAME_COUNT)
        #print("read..........")              
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if q.empty() ==True:
                #print('is not full.......')
                q.put(frame)     
            #else:
                #print("is full...")                       
        else:
            raise ValueError("No image!")
        
         
        
def view(q):
    yolo=YoloTest() 
    
         
    while True:
        frame =q.get()        	     
        image =  yolo.get_img(frame)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break        
            
def main():
    processes = []
    qu=mp.Queue(1)
     
        
    
    processes.append(mp.Process(target=read_img,args=(qu,)))
    processes.append(mp.Process(target=view,args=(qu,)))
    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()
        
if __name__ == '__main__':
    main()


