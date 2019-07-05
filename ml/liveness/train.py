from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import common as cn
from PIL import Image


vs = cv2.VideoCapture(cn.inf())
size = (int(vs.get(3)),int(vs.get(4)))
out = cv2.VideoWriter(cn.outf(), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),16.0,size)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(cn.sp_68)
face_rec = dlib.face_recognition_model_v1(cn.fr_path)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

MOUTH_THRESH=0.75
EYE_THRESH=0.15

def rend( shape,frame):
    hull = cv2.convexHull(shape)
    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

def face_pos_2(img ,sp,dist):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    if len (faces) == 1:
        _shape = predictor(gray, faces[0])
        face_desc = face_rec.compute_face_descriptor(img, _shape)
        dist.append(list(face_desc)) 

def face_pos(img,shape,dist):
    face_desc = face_rec.compute_face_descriptor(img, shape)
    dist.append(list(face_desc))


def train():
    times=0;
    dist=[]
    while True:
        return_value ,frame = vs.read()
        if return_value:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("No image!")

        rects = detector(gray, 0)
        if len(rects) == 1:
            _shape = predictor(gray, rects[0])
            shape = face_utils.shape_to_np(_shape)
            
            mouth = shape[mStart:mEnd]
            mar = cn.mouth_open(mouth)
            rend(mouth,frame)
            if mar > MOUTH_THRESH:
                cv2.putText(frame, "Mouth is Open!", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            left_eye = shape[lStart:lEnd]
            mar = cn.eye_close(left_eye)
            rend(left_eye,frame)
            if mar < EYE_THRESH :
                cv2.putText(frame, "left eye is close!", (30,90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)           
                cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            right_eye = shape[rStart:rEnd]
            mar = cn.eye_close(right_eye)
            rend(right_eye,frame)
            if mar < EYE_THRESH :
                cv2.putText(frame, "right eye is close!", (30,120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)           
                cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            ret, pitch, yaw, roll =cn.get_rolate_angle(frame,_shape,frame.shape)   
            angle_str = 'Y:{}, X:{}, Z:{}'.format(pitch, yaw, roll)
            cv2.putText( frame, angle_str, (20, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1 )
         

 

        else:
            print('not only one face')
            cv2.putText(frame, "not only one face!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if times%20 == 0:
            face_pos(frame,_shape,dist)
            if len(dist) == 2:
                dis=cn.face_dist(dist[0],dist[1])
                if dis > 0.45 :
                     print('not same people {:.2f} '.format(dis)) 
                cv2.putText(frame, "dist: {:.2f}".format(dis), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                _dist=[]
                _dist.append(dist[1])
                dist=_dist

        times+=1
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break;
      
    vs.stop()


train()      
         
