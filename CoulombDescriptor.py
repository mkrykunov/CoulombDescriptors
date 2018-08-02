#!/usr/bin/env python

import numpy as np
from numpy import linalg as LA
from collections import Counter


class CoulombDescriptor:

   def __init__(self, descriptor):
      self.descriptor = descriptor

   def ConstructDescriptor(self, InputMat, mat_Z = None):
      if self.descriptor == 'Coulomb2Eigval':
         return self.Coulomb2Eigval(InputMat)

      elif self.descriptor == 'XYZnuc2CoulombMat':
         return self.XYZnuc2CoulombMat(InputMat, mat_Z)

      elif self.descriptor == 'BagOfAtoms':
         Z_i = self.getNuclearCharges(InputMat)

         Atoms = self.CountAtoms(Z_i)

         Diag = self.Coulomb2Diagonal(InputMat)

         BoA = self.Diagonal2BoA(Diag, Z_i, Atoms)

         return BoA

      elif self.descriptor == 'BagOfBonds':
         Z_i = self.getNuclearCharges(InputMat)

         ZiZj = self.getOffdiagonal(Z_i)

         Bonds = self.CountBonds(ZiZj)

         BoB = self.Coulomb2BoB(InputMat, ZiZj, Bonds)

         return BoB
      elif self.descriptor == 'BagOfAtomsBonds':
         Z_i = self.getNuclearCharges(InputMat)

         Atoms = self.CountAtoms(Z_i)

         Diag = self.Coulomb2Diagonal(InputMat)

         BoA = self.Diagonal2BoA(Diag, Z_i, Atoms)

         ZiZj = self.getOffdiagonal(Z_i)

         Bonds = self.CountBonds(ZiZj)

         BoB = self.Coulomb2BoB(InputMat, ZiZj, Bonds)

         return np.concatenate((BoA, BoB), axis = 1)

   def get_CoulombMat(self, atom_xs, Zmat):
      m = len(Zmat)

      Cmat = np.zeros((m,m))

      S = np.dot(atom_xs, atom_xs.T)
      Sdiag = S.diagonal()
      D = np.sqrt(Sdiag[:,None] + Sdiag[None,:] - 2 * S)
      np.fill_diagonal(D, 1.0)

      Z_ij = np.outer(Zmat, Zmat)

      Cmat = np.divide(Z_ij, D)

      for i in range(m):
         Cmat[i,i] = 0.5 * np.power(Zmat[i], 2.4)

      return Cmat

   def sortDiagonal(self, mat_X):
      m = mat_X.shape[0]
      nx = mat_X.shape[1]

      X = np.zeros((m,nx,nx))

      for k in range(m):
         D = mat_X[k,:,:].diagonal()
         A = mat_X[k,:,:]

         sort_D = np.argsort(-D)
         sort_r_A = A[sort_D,:]
         sort_cr_A = sort_r_A[:,sort_D]

         X[k,:,:] = sort_cr_A

      return X

   def XYZnuc2CoulombMat(self, mat_R, Z, toSort = True):
      m = mat_R.shape[0]
      nx = mat_R.shape[1]

      X = np.zeros((m,nx,nx))

      for k in range(m):
         isize = nx - np.sum(Z[k,:] == 0)

         R_k = np.array(mat_R[k,0:isize,:], dtype = np.float64)

         A = self.get_CoulombMat(R_k, Z[k,0:isize])
         X[k,0:isize,0:isize] = A

      if toSort:
         X = self.sortDiagonal(X)

      return np.reshape(X, (m, nx * nx))

   def getNuclearCharges(self, mat_X):
      m = mat_X.shape[0]
      nx = mat_X.shape[1]

      Z_i = np.zeros((m,nx))

      for k in range(m):
         Z_i[k,:] = np.around(np.power(2.0 * mat_X[k,:,:].diagonal(), 5.0 / 12.0, dtype = np.float64))

      return Z_i

   def CountAtoms(self, Z_i):
      m = Z_i.shape[0]

      combined = Counter()

      for k in range(m):
         combined |= Counter(Z_i[k,:])

      Atoms = sorted(list(combined.elements()), reverse=True)

      Atoms = [a for a in Atoms if a != 0.0]

      return Atoms

   def Coulomb2Diagonal(self, mat_X):
      m = mat_X.shape[0]
      nx = mat_X.shape[1]

      D = np.zeros((m,nx))

      for k in range(m):
         D[k,:] = mat_X[k,:,:].diagonal()

      return D

   def Diagonal2BoA(self, Diagonal, Z_i, Atoms):
      m = Diagonal.shape[0]
      nx = len(Atoms)

      BoA = np.zeros((m,nx))

      for k in range(m):
         Z_k = Z_i[k,:]

         for zi in set(Z_k):
            if zi != 0.0:
               ind_A = np.where(Atoms == zi)[0]

               ind_Z = np.where(Z_k == zi)[0]

               BoA[k,ind_A[0:len(ind_Z)]] = sorted(Diagonal[k,ind_Z], reverse=True)

      return BoA

   def getOffdiagonal(self, Z_i):
      m = Z_i.shape[0]
      nx = Z_i.shape[1]

      ZiZj = {}

      for k in range(m):
         ZiZj[k] = [(Z_i[k,i],Z_i[k,j]) for i in range(nx) for j in range(i + 1,nx)]

      return ZiZj

   def CountBonds(self, ZiZj):
      combined = Counter()

      keys = sorted(ZiZj.keys())

      for k in keys:
         combined |= Counter(ZiZj[k])

      combined_list = [c for c in combined.elements()]

      Bonds = sorted(combined_list, reverse=True)

      Bonds = [b for b in Bonds if b[1] != 0.0]

      return Bonds

   def Coulomb2BoB(self, mat_X, ZiZj, Bonds):
      m = mat_X.shape[0]
      N = mat_X.shape[1]
      nx = len(Bonds)

      BoB = np.zeros((m,nx))

      for k in range(m):
         ZiZj_k = ZiZj[k]
         A = mat_X[k,:,:]
         U = A[np.triu_indices(N,1)]

         for zij in set(ZiZj_k):
            if zij[1] != 0.0:
               ind_B = [x for x, y in enumerate(Bonds) if y == zij]

               ind_Z = [x for x, y in enumerate(ZiZj_k) if y == zij]

               BoB[k,ind_B[0:len(ind_Z)]] = sorted(U[ind_Z], reverse=True)

      return BoB

   def Coulomb2Eigval(self, mat_X):
      m = mat_X.shape[0]
      nx = mat_X.shape[1]

      X = np.zeros((m,nx))

      for k in range(m):
         A = mat_X[k,:,:]
         D = LA.eigvalsh(A)
         D = np.reshape(D, (len(D), 1))
         D = -np.sort(-D, axis=0)
         X[k,:] = D[:,0]

      return X


