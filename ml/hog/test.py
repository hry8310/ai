import numpy as np
import cv2 as cv
from imutils.object_detection import non_max_suppression
 
 
hog = cv.HOGDescriptor()
hog.load('./log/hog.bin')
 
nn='test1.jpg'
npp='./dataset/test/'

 
img = cv.imread(npp+nn)
 
rects,scores = hog.detectMultiScale(img,winStride = (1,1),padding = (0,0),scale = 1.05)

print(len(rects))
 
sc = [score[0] for score in scores]
sc = np.array(sc)
 
for i in range(len(rects)):
    r = rects[i]
    rects[i][2] = r[0] + r[2]
    rects[i][3] = r[1] + r[3]
 
 
pick = []
print('rects_len',len(rects))
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
print('pick_len = ',len(pick))
 
for (x,y,xx,yy) in pick:
    cv.rectangle(img, (x, y), (xx, yy), (0, 0, 255), 2)    
 
cv.imwrite(npp+'o_'+nn, img)  
