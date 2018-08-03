#!/usr/bin/env python

import scipy.io as sio
import numpy as np
from CoulombDescriptor import CoulombDescriptor
from KernelRidgeRegression import KernelRidgeRegression


def test():
   mat = sio.loadmat('qm7.mat')

   mat_X = mat['X']
   mat_T = mat['T']
   mat_Z = mat['Z']
   mat_R = mat['R']
   mat_P = mat['P']

   desc = CoulombDescriptor('XYZnuc2CoulombMat')

   CoulombMat = desc.ConstructDescriptor(mat_R, mat_Z)

   desc2 = CoulombDescriptor('BagOfAtomsBonds')

   m = mat_Z.shape[0]
   nx = mat_Z.shape[1]

   CoulombMat = np.reshape(CoulombMat, (m, nx, nx))

   X = desc2.ConstructDescriptor(CoulombMat)

   print 'X.shape =', X.shape

   m0 = 200
   m1 = 100

   print 'm0 =', m0
   print 'm1 =', m1

   permut = np.random.permutation(mat_Z.shape[0])

   train = permut[0:m0]
   test = permut[m0:m0 + m1]

   X_train = X[train,:]
   t_train = mat_T[0,train]
   t_train = np.reshape(t_train, (len(t_train), 1))

   KRR_model = KernelRidgeRegression(kernel = 'Laplacian', sigma = 25000.0, lambd = 1.0E-8)

   y_train = KRR_model.fit_predict(X_train, t_train)

   err_train = np.sum(np.abs(y_train - t_train)) / len(t_train)

   print 'err_train = ', err_train

   X_test = X[test,:]
   t_test = mat_T[0,test]
   t_test = np.reshape(t_test, (len(t_test), 1))

   y_test = KRR_model.predict(X_train, X_test)

   err_test = np.sum(np.abs(y_test - t_test)) / len(t_test)

   print 'err_test = ', err_test

if __name__ == '__main__':
   test()
