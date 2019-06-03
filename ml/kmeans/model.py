from numpy import *
import time 

class Kmeans(object):
    def __init__(self,dataset,k,ker=1):
        if ker == 1:
            self.dataset=dataset
        else:
            self.dataset =array(self.kernel(dataset ,4))
        self.num,self.dim=self.dataset.shape
        print(self.dataset.shape)   
        self.ctid=zeros([k,self.dim])
        self.assIds=mat(zeros((self.num, 2)) )
        self.k=k
        self.mx=10000000
        sett={}
        self.ir=1
        for i in range(k):
            #idx= int(random.uniform(0, self.num)) 
            idx=self.rand(sett,self.num)
            self.ctid[i,:]=self.dataset[idx,:]
            self.assIds[idx,:]=i+1,0
        print('ssssssssssctid')
        print(self.ctid)

   
    def dis(self,v1,v2):
        return sum(power(v2 - v1, 2))

    def kernel(self,data, sigma):
        nData = len(data)
        Gram = [[0] * nData for i in range(nData)] # nData x nData matrix
        for i in range(nData):
            for j in range(i,nData):
                if i != j: # diagonal element of matrix = 0
                    # RBF kernel: K(xi,xj) = e ( (-|xi-xj|**2) / (2sigma**2)
                    square_dist = self.dis(data[i],data[j])
                    base = 2.0 * sigma**2
                    Gram[i][j] = exp(-square_dist/base)
                    Gram[j][i] = Gram[i][j]
        return Gram 
    
    def rand(self,sett,r):
        self.ir+=2
        return self.ir

    def rand1(self,sett,r):
        while True:
            idx= int(random.uniform(0, r)) 
            if idx not in sett.keys(): 
                sett[idx]=1
                return idx            
 
    def clutter(self):
        chd=True
    
        print(self.assIds)
        print('cccccccccccccccccccccccccccc')
        while chd:
            chd=False
            for i in range(self.num):
                minDis=self.mx
                minIdx= 0
                for j in range(self.k):
                    dis=self.dis(self.ctid[j,:],self.dataset[i,:])
                    if i==5 :
                       
                       print('dddddddddddddddddddddddddd %f' %dis)
                       #print(self.ctid[j,:] )
                       #print(self.dataset[i,:])
                    if dis<minDis:
                        minDis=dis
                        minIdx=j
                if self.assIds[i,0] != minIdx+1:
                    chd=True
                    self.assIds[i,:]=minIdx+1,minDis

            for j in range(self.k):
                cus=self.dataset[nonzero(self.assIds[:,0].A==j+1)[0]] 
                if len(cus) == 0:
                    print('llllllkkkkk  %d' %j)
                    print(self.assIds)
                self.ctid[j,:]=mean(cus,axis=0)
        print('yyyyyllllllkkkkk  %d' %j)
        print(self.assIds[:,0].A==5)
        print(nonzero(self.assIds[:,0].A==5))

