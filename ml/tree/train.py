import pandas as pd
from model import *
   
labels=[]
def data_set(file):
    lab=[]
    data=pd.read_csv(file,header=None)
    X=data.iloc[:].values
    for i in range(len(X[0])):
        lab.append('label'+str(i))
    print(labels )

    
    return X.tolist(),lab
    

trainset,labels=data_set('dataset.txt')
testset , _=data_set('testset.txt')    

tree=Tree()
tree.c45_run(trainset,labels,testset)

