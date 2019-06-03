from __future__ import division
import numpy as np

class classifier(object):

    def __init__(self):
        self.h=0.0001
    
    def _ap0(self,X,key):
       ret=[]
       for j in range(X.shape[0]):
           threshold=X[j,key]
           line=[]
           for k in range(X.shape[0]): 
               if X[k,key]>=threshold:
                   line.append(1)
               else:
                   line.append(-1)
           ret.append(line)
       return ret

    def _ap(self,X):
        ret={}
        for i in range(X.shape[1]):
            _ret= self._ap0(X,i)
            ret[i]=_ret 

        return ret

    def _e_key(self,ret,y,ws):
        err=0
        for i in range(len(ret)):
            if y[i] != ret[i]:
                err+=ws[i]
        return err

    def _e_keys(self,rets,y,ws):
        errs=[]
        for k in rets:
            errs.append(self._e_key(k,y,ws))

        return errs

    def e_keys(self,retss,y,ws):
        errs=[]
        for k in retss:
            errs.append(self._e_keys(retss[k],y,ws))

        return errs
 
    
    def _e_min(self,x):
        ret=10000
        for i in range(len(x)): 
            temp=min(x[i])
            if ret>temp :
                ret=temp

        for i in range(len(x)): 
            if ret==min(x[i]):
                return ret,i,x[i].index(ret)

    def clf(self,X,y,W):
        ey=self._ap(X)
        ex=self.e_keys(ey,y,W)
        self.e,self.key,self.idx=self._e_min(ex) 
        print(self.idx)
        self.h=X[self.idx][self.key]
        print('xxxxxxxxxxxxxxxxxxdddddddddddi   %f' %self.e)
        self.pred=ey[self.key][self.idx]

    def pd(self,X):
        if X[self.key]>=self.h:
           return 1
        return -1
