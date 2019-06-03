from math import log
import operator
#import treePlotter

class Tree(object):
    def __init__(self):
        print()
        self.trId=0

    def _gain(self,p):
        return p*log(p,2)
    

    def _ent(self,data):
        numEntries=len(data)
        labelCounts={}
        for featVec in data:
            currentlabel=featVec[-1]
            if currentlabel not in labelCounts.keys():
                labelCounts[currentlabel]=0
            labelCounts[currentlabel]+=1
        Ent=0.0
        for key in labelCounts:
            p=float(labelCounts[key])/numEntries
            Ent=Ent-self._gain(p)
        return Ent 

    def _feat(self,dataset,key):
         featlist=[example[key]for example in dataset]
         return set(featlist)

    def _splitdata(self,dataset,key,v):
        ret=[]
        for exa in dataset:
            if exa[key] == v:
                redexa=exa[:key]
                redexa.extend(exa[key+1:])                 
                ret.append(redexa)
        return ret
 
    def _feat_gain(self,dataset,key):
        uniqueVals=self._feat(dataset,key)
        for value in uniqueVals:
            subdataset=self._splitdata(dataset,key,value)   
            yield subdataset,value

    
    def chk_cls(self,dataset):
        cls=[ex[-1] for ex in dataset]
        cset=set(cls)
        if len(cset)==1:
            return cls[0]
        return None
            

    def max_feat(self,dataset):
        
        classCont={}        
        cls=[ex[-1] for ex in dataset]
        for vote in classList:
            if vote not in classCont.keys():
                classCont[vote]=0
            classCont[vote]+=1
        sortedClassCont=sorted(classCont.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCont[0][0]
  
    def chk_ret(self, dataset):
        cls= self.chk_cls( dataset )
        if cls != None :
            return cls

        if len(dataset[0]) == 1:
            return  self.max_feat(dataset)

        return None

    def _classify(self,tree,feat,test):
        firstStr = list(tree.keys())[0]
        secondDict = tree[firstStr]
        featIndex = feat.index(firstStr)        
        for key in secondDict.keys():
            if test[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    
                    classLabel,firstStr, key, tree = self._classify(secondDict[key], feat, test)
                else :
                    classLabel=secondDict[key]
        return classLabel ,firstStr,key,tree
               
    def classify(self,tree,feats,tests):
        labels= []
        for test in tests:
 
            label,feat,key,dic =self._classify(tree,feats,test)
            labels.append( label)
        return labels           

    def dev(self ,tree,feats,tests):
        pa={}
        for i in range(len(tests)):
            test=tests[i][0:-1]
            label,feat,key,tr =self._classify(tree,feats,test)        
            lf={}
            print(tr)
            if tr['trId'] not in pa.keys():
                pa[tr['trId']]=lf
            else:
                lf=pa[tr['trId']]
            
            lfd={'all_exm':0,'rgt_exm':0} 
            
            if key not in lf.keys():
                lf[key]=lfd
            else:
                lfd=lf[key]
          
            lfd['all_exm']+=1
            if label==tests[i][-1] :
                lfd['rgt_exm']+=1
        print(pa)
        return pa


             
             
                 
                 


    def _id3_gain(self,dataset):
        num = len(dataset[0]) - 1
        baseEnt=self._ent(dataset)
        bestGain=0.0
        bestFeat=1
        
        for i in range(num):
            newEnt=0.0
            for sub,vlaue in self._feat_gain(dataset,i):
                p=len(sub)/float(len(dataset)) 
                newEnt+=p*self._ent(sub)    
            infoGain=baseEnt-newEnt
            if infoGain>bestGain:
                bestGain=infoGain
                bestFeat=i
        return bestFeat
    

    def id3_tree(self,dataset,label):
        cls=self.chk_ret(dataset)
        if cls != None:
            return cls

        bestFeat= self._id3_gain(dataset)
        bestFeatLabel = label[bestFeat]
        self.trId+=1
        ID3Tree = {bestFeatLabel:{},trId:self.trId}
        del(label[bestFeat])
        uv=self._feat(dataset,bestFeat) 
   
        for v in uv :
            sublabel=label[:]
            ID3Tree[bestFeatLabel][v]=self.id3_tree(self._splitdata(dataset,bestFeat,v),sublabel)
        return ID3Tree

    def id3_run(self,dataset,labels,testset):
        
        labels_tmp = labels[:]
        ID3Tree = self.id3_tree(dataset,labels_tmp)
        cls=self.classify(ID3Tree, labels, testset) 
        print('xxxxxxxxxxxxxxxxxxx', cls)


    def _c45_gain(self,dataset):
        num = len(dataset[0]) - 1
        baseEnt=self._ent(dataset)
        bestGain=0.0
        bestFeat=1
        
        for i in range(num):
            newEnt=0.0
            hgain=0.0
            for sub,vlaue in self._feat_gain(dataset,i):
                p=len(sub)/float(len(dataset)) 
                newEnt+=p*self._ent(sub)    
                hgain=hgain-self._gain(p)
            if hgain==0:
                continue
            gainRatio=(baseEnt-newEnt)/hgain
            if gainRatio>bestGain:
                bestGain=gainRatio
                bestFeat=i
        return bestFeat
    


    def c45_tree(self,dataset,label):
        cls=self.chk_ret(dataset)
        if cls != None:
            return cls

        bestFeat= self._c45_gain(dataset)
        bestFeatLabel = label[bestFeat]
        C45Tree = {bestFeatLabel:{},'trId':self.trId }
        self.trId+=1
        del(label[bestFeat])
        uv=self._feat(dataset,bestFeat) 
   
        for v in uv :
            sublabel=label[:]
            C45Tree[bestFeatLabel][v]=self.c45_tree(self._splitdata(dataset,bestFeat,v),sublabel)
        return C45Tree

    def c45_run(self,dataset,labels,testset):
        labels_tmp = labels[:]
        self.trId=0
        c45Tree = self.c45_tree(dataset,labels_tmp)
        # cls=self.classify(c45Tree, labels, testset) 
        #  print('xxxxxxxxxxxxxxxxxxx', cls)
        self.dev(c45Tree,labels,dataset)

    def gini( self,g,p):
        g+=p*(1-p)
        return g
 
    def _gini(self,data):
        numEntries=len(data)
        labelCounts={}
        for featVec in data:
            currentlabel=featVec[-1]
            if currentlabel not in labelCounts.keys():
                labelCounts[currentlabel]=0
            labelCounts[currentlabel]+=1
        g=0.0
        for key in labelCounts:
            p=float(labelCounts[key])/numEntries
            g=self.gini(g,p)
        return g 

            


    def _cart_gain(self,dataset):
        num = len(dataset[0]) - 1
        bestGini=99999.9
        bestFeat=-1
        
        for i in range(num):
            hgini=0.0
            for sub,vlaue in self._feat_gain(dataset,i):
                p=len(sub)/float(len(dataset)) 
                hgini+=p*self._gini(sub)    
            if hgini<bestGini:
                bestGini=hgini
                bestFeat=i
        return bestFeat
    


    def cart_tree(self,dataset,label):
        cls=self.chk_ret(dataset)
        if cls != None:
            return cls

        bestFeat= self._cart_gain(dataset)
        bestFeatLabel = label[bestFeat]
        cartTree = {bestFeatLabel:{}}
        del(label[bestFeat])
        uv=self._feat(dataset,bestFeat) 
   
        for v in uv :
            sublabel=label[:]
            cartTree[bestFeatLabel][v]=self.cart_tree(self._splitdata(dataset,bestFeat,v),sublabel)
        return cartTree

    def cart_run(self,dataset,labels,testset):
        labels_tmp = labels[:]
        cartTree = self.cart_tree(dataset,labels_tmp)
        cls=self.classify(cartTree, labels, testset) 
        print('xxxxxxxxxxxxxxxxxxx', cls)


