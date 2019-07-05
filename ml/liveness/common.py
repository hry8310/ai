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
import math
from PIL import Image

sp_68='/root/dlib/shape_predictor_68_face_landmarks.dat'
fr_path = "/root/dlib/dlib_face_recognition_resnet_model_v1.dat"


video_name='test3.mp4'
inp='./inputs/'
outp='./outputs/'





def inf(name=video_name):
    video_name=name
    return inp+name

def outf():
    return outp+'o_'+video_name

def eye_close(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4]) 

    C = dist.euclidean(eye[0], eye[3]) 

    mar = (A + B) / (2.0 * C)

    return mar


def mouth_open(mouth):
    A = dist.euclidean(mouth[2], mouth[10]) 
    B = dist.euclidean(mouth[4], mouth[8]) 

    C = dist.euclidean(mouth[0], mouth[6])

    mar = (A + B) / (2.0 * C)

    return mar

def face_dist(dist_1,dist_2):
    dis = np.sqrt(sum((np.array(dist_1)-np.array(dist_2))**2))
    return dis

def face_cor(shape):
    if shape.num_parts != 68 :
        return None
    points = np.array([
        (shape.part(30).x, shape.part(30).y),     # Nose 
        (shape.part(8).x,  shape.part(8).y),      # Chin
        (shape.part(36).x, shape.part(36).y),     # Left eye 
        (shape.part(45).x, shape.part(45).y),     # Right eye 
        (shape.part(48).x, shape.part(48).y),     # Left Mouth 
        (shape.part(54).x, shape.part(54).y)      # Right mouth 
    ], dtype="double")
    return points

def get_camera_point(img_size, image_points ):
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose 
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye 
        (225.0, 170.0, -135.0),      # Right eye 
        (-150.0, -150.0, -125.0),    # Left Mouth 
        (150.0, -150.0, -125.0)      # Right mouth 
    ])
    
    
    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    cam_mat = np.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype = "double"
    )
    
    
    _dist = np.zeros((4,1)) # Assuming no lens distortion
    (succ, r_v, t_v) = cv2.solvePnP(model_points, image_points, cam_mat, _dist, flags=cv2.SOLVEPNP_ITERATIVE )
    return succ, r_v, t_v, cam_mat, _dist


def _get_rolate_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    
    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2)*rotation_vector[0][0] / theta
    y = math.sin(theta / 2)*rotation_vector[1][0] / theta
    z = math.sin(theta / 2)*rotation_vector[2][0] / theta
    
    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    pitch = math.atan2(t0, t1)
    
    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)
    
    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)
    
    
	# 单位转换：将弧度转换为度
    Y = int((pitch/math.pi)*180)
    X = int((yaw/math.pi)*180)
    Z = int((roll/math.pi)*180)
    
    return Y, X, Z

def get_rolate_angle(img,shape, im_szie):
    image_points = face_cor(shape)
    if image_points is None:
        print('get_image_points failed')
        return -1, None, None, None
    
    ret, r_v, t_v, cam_mat, _dist = get_camera_point(im_szie, image_points)
    if ret != True:
        print(' get_camera_point failed')
        return -1, None, None, None
    
    pitch, yaw, roll = _get_rolate_angle(r_v)

    return 0, pitch, yaw, roll
    
        
def get_rolate_angle_debug(frame,shape, im_szie):
    image_points = face_cor(shape)
    if image_points is None:
        print('get_image_points failed')
        return -1, None, None, None
    
    ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_camera_point(im_szie, image_points)
    if ret != True:
        print(' get_camera_point failed')
        return -1, None, None, None
   
    pitch, yaw, roll = _get_rolate_angle(rotation_vector)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
    for p in image_points:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
         
         
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
         
    cv2.line(frame, p1, p2, (255,0,0), 2)

    return 0, pitch, yaw, roll
    
        
