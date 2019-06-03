import pandas as pd
from model import *
from six.moves import cPickle

data=pd.read_csv('data.csv')

#get X and y
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#train the AdaboostClassifier
clfer=adaboost()
times=clfer.clf(X,y)
with open('ada.md', 'wb') as f:
    cPickle.dump(clfer, f)

with open('ada.md', 'rb') as f:
    clff=cPickle.load(f)

clff.run_pred(X[2])
