from numpy import *
import operator as op 

class Knn(object):
    def __init__(self):
        print()

    def dis(self,v1,v2):
        return sqrt(sum(power(array(v2) - array(v1), 2))) 

    def find_nb(self,diss,k):
        diss.sort(key=op.itemgetter(len(diss[0])-1))
        return diss[0:k] 

    def find_sm(self ,diss):
        vote={}
        for one in diss:
            c = one[-2] 
            if c not in vote.keys():
                 vote[c]=0;
            vote[c]+=1
        svote=sorted(vote.items(),key=op.itemgetter(1),reverse=True)
        return svote[0][0]
   
    def _knn(self,dataset, test,k):
        diss=[]
        for data in dataset:
            #data=data.tolist()
            dis=self.dis(data[:-1], test)
            diss.append(data+[dis])
        _diss=self.find_nb(diss,k)
        c=self.find_sm(_diss)
        print(c)
        return c           
 
     
    def knn(self,dataset,testset,k):
        for test in testset:
            self._knn(dataset,test,k) 
