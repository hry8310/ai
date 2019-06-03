import pandas as pd
from model import *

data=pd.read_csv('seed.txt',sep=' ')

#get X and y
dataset=data.iloc[:].values
print('dataset.....................')

#train the AdaboostClassifier
kmeans=Kmeans(dataset,10,1)

kmeans.clutter()
