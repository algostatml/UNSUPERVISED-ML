#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:08:33 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np

class GMM(object):
    def __init__(self, k):
        if not k:
            k = 2
            self.k = k
        else:
            self.k = k
        return
    
    def fit(self, X, iteration):
        '''
        '''
        if not iteration:
            iteration = 100
            self.iteration = iteration
        else:
            self.iteration = iteration
        return
    
    def predict(self, X):
        '''
        :param: X: NxD
        '''
        return
        