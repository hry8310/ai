import pandas as pd
from model import *

data=pd.read_csv('dataset.csv')

#get X and y
X=data.iloc[:].values.tolist()
data=pd.read_csv('test.csv')
Y=data.iloc[:].values.tolist()
#train the AdaboostClassifier
knn=Knn()
knn.knn(X,Y,8)
