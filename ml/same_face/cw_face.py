import sys
import os
import dlib
import pickle
import glob
import cv2
import numpy as np
from imutils import build_montages



predictor_path = '/root/dlib/shape_predictor_5_face_landmarks.dat'
face_rec_model_path = '/root/dlib/dlib_face_recognition_resnet_model_v1.dat'
img_path = 'all' 
output_path = 'cwout' 
enc_path='./pkl/cwface.pkl'

detector = dlib.get_frontal_face_detector() #a detector to find the faces
sp = dlib.shape_predictor(predictor_path) #shape predictor to find face landmarks
facerec = dlib.face_recognition_model_v1(face_rec_model_path) #face recognition model


def face_cfg():
    if os.path.exists(enc_path):
        data = pickle.loads(open(enc_path, "rb").read())
        return data

    imgs=[]
    desc=[]
    for f in glob.glob(os.path.join(img_path, "*.jpg")):
        print('process img {}'.format(f))
        img = dlib.load_rgb_image(f)

        dets = detector(img, 1)

        for k, d in enumerate(dets):
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)

            face_descriptor = facerec.compute_face_descriptor(img, shape)
            desc.append(face_descriptor)
            imgs.append((f, shape,d))
    data={}
    data['imgs']=imgs
    data['desc']=desc
    f = open(enc_path, "wb")
    f.write(pickle.dumps(data))
    f.close()
    return data

def cw_face(data):
    desc=data['desc']
    imgs=data['imgs']
    labels = dlib.chinese_whispers_clustering(desc, 0.5)
    num_classes = len(set(labels)) 
    print("all class: {}".format(num_classes))

    for i in range(0, num_classes):
        indices = []
        class_length = len([label for label in labels if label == i])
        for j, label in enumerate(labels):
            if label == i:
                indices.append(j)
        faces=[] 
        for k, index in enumerate(indices):
            f, shape,d = imgs[index]
            img =cv2.imread(f)
            face=cropImg(img,d)
            face = cv2.resize(face, (96, 96))
            faces.append(face)
        montage = build_montages(faces, (96, 96), (5, 5))[0]
        cv2.imwrite('./whout/lab_'+str(i)+'_img.jpg', montage)


def cropImg(img, det):
    
    l = float(det.left())
    t = float(det.top())
    ww = float(det.right() - l)
    hh = float(det.bottom() - t)

    h = img.shape[0]
    w = img.shape[1]
    cx = l + ww/2
    cy = t + hh/2
    tsize = max(ww, hh)/2
    l = cx - tsize
    t = cy - tsize

    bl = int(round(cx - 1.1*tsize))
    bt = int(round(cy - 1.1*tsize))
    br = int(round(cx + 1.1*tsize))
    bb = int(round(cy + 1.1*tsize))
    nw = int(br - bl)
    nh = int(bb - bt)
    imcrop = np.zeros((nh, nw, 3), dtype='uint8')

    ll = 0
    if bl < 0:
        ll = -bl
        bl = 0
    rr = nw
    if br > w:
        rr = w+nw - br
        br = w
    tt = 0
    if bt < 0:
        tt = -bt
        bt = 0
    bbb = nh
    if bb > h:
        bbb = h+nh - bb
        bb = h
    imcrop[tt:bbb,ll:rr,:] = img[bt:bb,bl:br,:]
    return imcrop


data=face_cfg()
cw_face(data)    


