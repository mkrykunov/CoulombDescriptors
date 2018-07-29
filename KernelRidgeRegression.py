#!/usr/bin/env python

import numpy as np
from numpy import linalg as LA


class KernelRidgeRegression:

   def __init__(self, kernel, sigma, lambd):
      self.kernel = kernel
      self.sigma = sigma
      self.lambd = lambd

   def fit_predict(self, X, t):
      if self.kernel == 'Gaussian':
         K = self.GaussianKernel(X)
      elif self.kernel == 'Laplacian':
         K = self.LaplacianKernel(X)

      self.alpha = np.dot(LA.pinv(K + self.lambd * np.eye(len(t))), t)

      y = np.dot(K, self.alpha)

      return y

   def predict(self, X, X1):
      if self.kernel == 'Gaussian':
         K1 = self.GaussianKernel2(X1, X)
      elif self.kernel == 'Laplacian':
         K1 = self.LaplacianKernel2(X1, X)

      y1 = np.dot(K1, self.alpha)

      return y1

   def GaussianKernel(self, X):
      m = X.shape[0]
      K = np.zeros((m,m))

      const = 1.0 / (2 * self.sigma**2)

      S = np.dot(X, X.T)
      Sdiag = S.diagonal()
      K0 = Sdiag[:,None] + Sdiag[None,:] - 2 * S
      K = np.exp(-const * K0)

      return K

   def LaplacianKernel(self, X):
      m = X.shape[0]
      K = np.zeros((m,m))

      const = 1.0 / self.sigma

      for i in range(m):
         tmp = np.exp(-const * np.sum(np.abs(X[i,:] - X[:,:]), axis=1))
         K[i,:] = tmp

      return K

   def GaussianKernel2(self, X, X1):
      m = X.shape[0]
      m1 = X1.shape[0]
      K = np.zeros((m,m1))

      const = 1.0 / (2 * self.sigma**2)

      for i in range(m):
         tmp = np.exp(-const * np.sum(np.square(X[i,:] - X1[:,:]), axis=1))
         K[i,:] = tmp

      return K

   def LaplacianKernel2(self, X, X1):
      m = X.shape[0]
      m1 = X1.shape[0]
      K = np.zeros((m,m1))

      const = 1.0 / self.sigma

      for i in range(m):
         tmp = np.exp(-const * np.sum(np.abs(X[i,:] - X1[:,:]), axis=1))
         K[i,:] = tmp

      return K

