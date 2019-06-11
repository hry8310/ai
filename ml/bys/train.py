from model import Bayes
import pandas as pd

class Test(object):
    def __init__(self):
        self.md=Bayes()
        print()
     
 
    def gen_vocab(self,dataset):
        self.vocabSet= set([])        
        for document in dataset:
            self.vocabSet = self.vocabSet | set(document)
        self.vocab=list(self.vocabSet)
        
    def toVer(self,data):
        v=[0]*len(self.vocab)
        for w in data:
            if w in self.vocab:
                v[self.vocab.index(w)]+=1
        return v

    def train(self,dataset,labels):
        self.gen_vocab(dataset)
        tds=[]
        for d in dataset:
            tds.append(self.toVer(d))

        self.md.train(tds,labels) 

    def clf(self,tss):
        ts=self.toVer(tss)
        self.md.clf(ts)



def loadDataSet():
    X=[]
    Y=[]
    with open('data.txt', 'r') as openFile:
        data = openFile.readlines()
        for line in data:
            _l= line.strip().split(',')
            X.append(_l[:-1])
            Y.append(_l[-1])
    return X,Y

def loadTestSet():
    with open('test.txt', 'r') as openFile:
        data = openFile.readlines()
        for line in data:
            return line.strip().split(',')

dataset,labels=loadDataSet()
test=Test();
test.train(dataset,labels)
tss=loadTestSet()
test.clf(tss)

