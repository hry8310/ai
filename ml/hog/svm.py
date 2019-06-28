import numpy as np
import cv2 as cv
 
#svm参数配置
class SVM(object):
    def __init__(self):
        svm = cv.ml.SVM_create()
        svm.setCoef0(0)
        svm.setCoef0(0.0)
        svm.setDegree(3)
        criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 1000, 1e-3)
        svm.setTermCriteria(criteria)
        svm.setGamma(0)
        svm.setKernel(cv.ml.SVM_LINEAR)
        svm.setNu(0.5)
        svm.setP(0.1)
        svm.setC(0.01)
        svm.setType(cv.ml.SVM_EPS_SVR)
        self.svm=svm 
 
    def train(self,feats,labels):
        self.svm.train(np.array(feats),cv.ml.ROW_SAMPLE,np.array(labels))
    
    def svm_save(self,name):
        self.svm.save(name)
        
    def svm_load(self,name):
        svm = cv.ml.SVM_load(name)
    
        return svm

