import numpy as np
import cv2 as cv
 
from read import get_files
from svm import SVM
 
class Hog(object):
    def __init__(self):
        self.svm=SVM()    
        self.hog=cv.HOGDescriptor()



   
     
    def _feat(self,imgs,label,wsize = (64,128)):
        features=[]
        for i in range(len(imgs)):
            roi = cv.resize(imgs[i] ,wsize) 
            features.append(self.hog.compute(roi))
        
        return features,label
    
     
    def get_svm_detector(self):
        sv = self.svm.svm.getSupportVectors()
        rho, _, _ = self.svm.svm.getDecisionFunction(0)
        sv = np.transpose(sv)
        isv= np.append(sv,[[-rho]],0)        
        return isv
     
    def feat(self):
        imgs,labels= get_files()
        f,labels=self._feat(imgs,labels)
        return f,labels
    
     
    #hog训练
    def train(self):
        
        #get hog features
        features,labels=self.feat()
        
        #svm training
        print ('svm training...')
        #print(labels)
        self.svm.train(features,labels)
        print ('svm training complete...')
        
        _sv= self.get_svm_detector()
        print(_sv.shape)
        self.hog.setSVMDetector(_sv)
        self.hog.save('./log/hog.bin')
    
        return self.hog    
       
    def test(self):
        nn='test1.jpg'
        npp='./dataset/test/'
    
    
        img = cv.imread(npp+nn)
    
        rects,scores = self.hog.detectMultiScale(img,winStride = (1,1),padding = (0,0),scale = 1.05)
       
        print(len(rects))
     
        

