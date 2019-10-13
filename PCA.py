#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 08:05:26 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np


class PCA:
    def __init__(self, k = None):
        if not k:
            k = 2
            self.k = k
        else:
            self.k = k
        return
    
    def explained_variance_(self):
        '''
        :Return: esplained variance.
        '''
        self.total_variance = self.cov.diagonal()
        self.explained_variance = 
        return 
    
    def fit(self, X):
        self.X = X
        self.X = self.X - np.mean(self.X, axis = 0)
        self.cov = (1/self.X.shape[1])* np.dot(self.X.T, self.X)
        self.eival, self.eivect = np.linalg.eig(self.cov)
        #sort eigen values
        self.explained_variance = sorted(self.eival[:self.k], reverse = True)
        self.eival, self.eivect = self.eival[:self.k], self.eivect[:, :self.k]
        return self
    
    def fit_transform(self):
        return self.X.dot(self.eivect[:, :self.k])

#%% Testing
from sklearn.datasets import load_iris
X, y = load_iris().data, load_iris().target
A = np.array([[1, 2], [3, 4], [5, 6]])
pca = PCA(k = 2).fit(X)
newX = pca.fit_transform()

per_var = np.round(pca.explained_variance * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('percentange of explained variance')
plt.xlabel('principal component')
plt.title('scree plot')
plt.show()



