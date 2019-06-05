#!/usr/bin/env python

from model import SVM
from kernel import Kernel

import logging
import numpy as np
import pandas as pd

data=pd.read_csv('data.csv')

d_X=data.iloc[:,:-1].values
d_y=data.iloc[:,-1].values
d_y=d_y.reshape([-1,1])

def test():
    samples = np.matrix(d_X)
    labels=np.matrix(d_y) 
    svm = SVM(Kernel._polykernel(4,10.0), 1)
    predictor = svm.train(samples, labels)
    if predictor is None :
        print('can not classify')
        return
    test = np.matrix( d_X[15])

    y=predictor.predict(test)
    print(y)

test()
