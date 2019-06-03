import numpy as np
from classifer import *


class adaboost(object):
    def __init__(self):
        self.max_step=100

    def alpha(self,e):
        if e==0:
            return 10000
        elif e==0.5:
            return 0.001
        else:
            return 0.5*np.log((1-e)/e)


    def __W(self,w,a,y,G):
        return w*np.exp(-1*a*y*G)

    def _Z(self, weights,a,ys,Gs):

        return sum(self.__W(weights[i],a,ys[i],Gs[i]) for i in range(len(ys)))

 
    def new_w(self,weights,a,ys,Gs,Z): 
        _w=[]
        for i in range(len(ys)):
            _w.append(self.__W(weights[i],a,ys[i],Gs[i])/Z)
        return np.array(_w)
 
    def final_p(self,i,a,clfs,y):
        ret=np.array([0.0]*len(y))
        print(a) 
        for j in range(i+1):
            print(a[j])
            print(clfs[j].pred)
            ret+=np.array(a[j])*clfs[j].pred                
        return np.sign(ret)
  
    def final_e(self,y,cal_final_predict): 
        ret=0
        for i in range(len(y)):
            if y[i]!=cal_final_predict[i]:
                ret+=1
        return ret/len(y)
    
    def clf(self,X,y,M=15): 
        self.W={}
        self.clfs={}
        self.a={}
        self.pred={}

        for i in range(M):
            self.W.setdefault(i)
            self.clfs.setdefault(i)
            self.a.setdefault(i)
            self.pred.setdefault(i)
      
        for i in range(M):
            if i == 0:
                self.W[i]=np.array([1]*len(y))/len(y) 
                self.W[i].reshape([len(y),1])

            else:
                z=self._Z(self.W[i-1],self.a[i-1],y,self.pred[i-1] )
                self.W[i]=self.new_w(self.W[i-1],self.a[i-1],y,self.pred[i-1],z )

            clfier=classifier()
            clfier.clf(X,y,self.W[i])
            self.clfs[i]=clfier
            self.a[i]=self.alpha(clfier.e)            
            self.pred[i]=clfier.pred
            
            er=self.final_p(i,self.a,self.clfs,y)

            ee=self.final_e(y,er)
            if ee==0 or clfier.e==0:
                break 


    def run_pred(self,X):
        sum=0.0
        print(self.clfs) 
        for k in self.clfs.keys():
            
            clf=self.clfs[k]
            if clf is None:
                break
            ret=clf.pd(X)
            print(ret)
            sum=sum+ret*self.a[k]
        print(sum)
        if sum >= 0.0:
            return 1
        return -1
 
