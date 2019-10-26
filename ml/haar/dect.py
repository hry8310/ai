import numpy as np
import cv2

cadexml='/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_default.xml'
cadexml='/root/mt/face2/cascade/cascade.xml'
img_name='c1.jpg'
i_path='./test/'+img_name
o_path='./out/o_'+img_name


cfer = cv2.CascadeClassifier(cadexml)

img = cv2.imread(i_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
boxes=cfer.detectMultiScale(gray)
print(boxes)
for (x,y,w,h) in boxes:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imwrite(o_path,img)
