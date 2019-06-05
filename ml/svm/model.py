import numpy as np
import cvxopt.solvers
import logging


MIN_SVM = 1e-5


class SVM(object):
    def __init__(self, kernel, c):
        self._kernel = kernel
        self._c = c

    def train(self, X, y):
        a_s = self._cvx_op(X, y)
        return self._pred(X, y, a_s)

    def _kernel_mat(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def _pred(self, X, y, a_s):
        a_i = a_s > MIN_SVM
 
        a = a_s[a_i]
        print(len(a))
        x_v = X[a_i]
        y_v = y[a_i]
        if len(a) == 0 :
            print('un split')
            return 

        b = np.mean(
            [y_k - Predictor(
                kn=self._kernel,
                b=0.0,
                a=a,
                x=x_v,
                y=y_v).predict(x_k)
             for (y_k, x_k) in zip(y_v, x_v)])

        return Predictor(
            kn=self._kernel,
            b=b,
            a=a,
            x=x_v,
            y=y_v)

    def _cvx_op(self, X, y):
        n_samples, _ = X.shape
        K = self._kernel_mat(X)
        #  according to zzh-tong ji xue xi fang fa 

        ## EiEjai*aj*yi*yj*K(xi,xj) 
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # a>=0
        G_G_0 = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_0 = cvxopt.matrix(np.zeros(n_samples))

        # a<=c
        G_L_C = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_c = cvxopt.matrix(np.ones(n_samples) * self._c)
       
        #union
        G = cvxopt.matrix(np.vstack((G_G_0, G_L_C)))
        h = cvxopt.matrix(np.vstack((h_0, h_c)))
       
        #Eai*yi=0 
        AY = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)
        solution = cvxopt.solvers.qp(P, q, G, h, AY, b)

        return np.ravel(solution['x'])


class Predictor(object):
    def __init__(self,
                 kn,
                 b,
                 a,
                 x,
                 y):
        self._kn = kn
        self._b = b
        self._a = a 
        self._x = x
        self._y= y 

    def predict(self, xj):
        #yj=Eyi*ai*k(xi.xj)+b according to p127
        result = self._b
        for a_i, x_i, y_i in zip(self._a,
                                 self._x,
                                 self._y):
            result += a_i * y_i * self._kn(x_i, xj)
        return np.sign(result).item()
