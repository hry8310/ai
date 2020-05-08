import dlib
import numpy as np
import cv2
import os




def recognition(img):
    dets = detector(img, 1)
    bb = np.zeros(4, dtype=np.int32)
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

        bb[0] = np.maximum(d.left(), 0)
        bb[1] = np.maximum(d.top(), 0)
        bb[2] = np.minimum(d.right(), img.shape[1])
        bb[3] = np.minimum(d.bottom(), img.shape[0])
        rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
        shape = sp(img, rec)
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        cv2.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 255, 0), 2)
    
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
     
    cv2.imshow("result", img)
    cv2.waitKey(1)
 

video_path      = "http://admin:admin@192.168.1.102:8081"
def main():
    vid = cv2.VideoCapture(video_path)
    while True:
        return_value, frame = vid.read()
        recognition(frame)



predictor_path = "E:/ai/dlib/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "E:/ai/dlib/dlib_face_recognition_resnet_model_v1.dat"
detector = dlib.get_frontal_face_detector()
sp= dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

if __name__ == '__main__':
    main()
