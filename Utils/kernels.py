#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:11:48 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np


class Kernels:
    '''Docstring
    Kernels are mostly used for solving
    non-lineaar problems. By projecting/transforming
    our data into a subspace, making it easy to
    almost accurately classify our data as if it were
    still in linear space.
    '''
    def __init__(self):
        return
    
    @staticmethod
    def linear(x1, x2, c = None):
        '''
        Linear kernel
        ----------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :return type: kernel(Gram) matrix
        '''
        if not c:
            c = 0
        return x1.dot(x2.T) + c
    
    @staticmethod
    def rbf(x1, x2, gamma = None):
        '''
        RBF: Radial basis function or guassian kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 11 #we are using a large gamma for the make_moon dataset
        if x1.ndim == 1 and x2.ndim == 1:
            return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
            return np.exp(-gamma * np.linalg.norm(x1 - x2, axis = 1)**2)
        elif x1.ndim > 1 and x2.ndim > 1:
            return np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 2)**2)
        
    @staticmethod
    def sigmoid(x1, x2, gamma = None, c = None):
        '''
        logistic or sigmoid kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 10
        if not c:
            c = 1
        return np.tanh(gamma * x1.dot(x2.T) + c)
    
    @staticmethod
    def polynomial(x1, x2, d = None):
        '''
        polynomial kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: d: polynomial degree
        :return type: kernel(Gram) matrix
        '''
        if not d:
            d = 3
        return (x1.dot(x2.T))**d
    
    @staticmethod
    def cosine(x1, x2):
        '''
        Cosine kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :return type: kernel(Gram) matrix
        '''
        
        return (x1.dot(x2.T)/np.linalg.norm(x1, 1) * np.linalg.norm(x2, 1))
    
    @staticmethod
    def correlation(x1, x2, gamma = None):
        '''
        Correlation kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 10
        return np.exp((x1.dot(x2.T)/np.linalg.norm(x1, 1) * np.linalg.norm(x2, 1)) - gamma)
    
    
    
