import numpy as np
import operator
from functools import reduce

class Bayes(object):
    def __init__(self):
        self.cnum={}
        self.cls={}
        self.vs={}
        self.vcc={}

    def c_p(self,labels):
        for i in range(len(labels)) :  
           c=labels[i]
           if c not in self.cnum:
               self.cnum[c]=1
           else:
               self.cnum[c] +=1 
        for c in self.cnum.keys():
            self.cls[c]=self.cnum[c]/len(labels) 
 

    def p_v(self):
        self.vp={}
        for k in self.vs.keys():
            self.vp[k]=np.log(self.vs[k]/self.vcc[k])


    def train(self, dataset,labels):
        cn=len(labels)
        dl=len(dataset[0])
        for i in range(cn):
            c=labels[i] 
            if c not in self.vs :
                self.vs[c]=np.ones(dl)
                self.vcc[c]=2;
                
            self.vs[c]=self.vs[c]+dataset[i]
            self.vcc[c]=self.vcc[c]+sum(dataset[i])

        self.c_p(labels)
        self.p_v()
  

    def clf(self,tests):
        tss={}
        for k in self.vp.keys():
            tss[k]=sum(tests*self.vp[k])+np.log(self.cls[k])
        sotss=sorted(tss.items(),key=operator.itemgetter(1),reverse=True)
        print(sotss)
        print(sotss[0][0])
        return sotss[0][0]




