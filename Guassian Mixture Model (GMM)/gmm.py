# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:10:12 2020

@author: ifeanyi.ezukwoke
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import inv, det

class GMM(object):
    '''Docstring
    Guassian Mixture Model
    ------------------------
    
    Parameters:
    ----------------
        k: numbers of mixture components
        
    Attributes
    ------------------
    k: numbers of cluster
    eps: convergence threshold
    
    Return
    ------------------
    
    
    '''
    def __init__(self, k):
        '''
        Parameters
        -----------
        k: cluster number
        eps: convergence threshold or the tolerance 
        '''
        self.k = k
        
        return
    
    @staticmethod
    def logLikelihood(X, clusters):
        '''No need to multiple by responsible in Q since they always sum to 1
        '''
        log_lhd = np.log(np.array([clust['denom'] for clust in clusters])) #log likelihood for x_n
        return np.sum(log_lhd)
    
    @staticmethod
    def mvn(X, mu, cov):
        '''Multivariate Normal distribution
        
        Parameters
        -----------
        X: data set
        mu: mean of vectors in X
        cov: covariance matrix of X
        
        Return type
        ------------
        probability of distribution (NX1 - dimension array)
        '''
        n, m = X.shape
        u = (X - mu)
        f_t = 1/((2*np.pi)**(m/2) * det(cov)**.5) # first term 
        s_t = np.exp(-0.5 * u @ inv(cov) @ u.T) #second term
        return np.diagonal(f_t*s_t).reshape(-1, 1)
    
    @staticmethod
    def initialize_clusters(X, k):
        n, m = X.shape
        clusters = []
        # We use the KMeans centroids to initialise the GMM
        kmeans = KMeans(k).fit(X)
        mu_k = kmeans.cluster_centers_
        
        for ii in range(k):
            clusters.append({'pi_k': 1.0 / k,\
                             'mu_k': mu_k[ii],\
                             'cov_k': np.identity(m, dtype = np.float64)
            })
            
        return clusters
    
    
    def expectation(self, X, clusters):
        '''Compute the responsibility for individual clusters
        '''
        n, m = X.shape
        denom = np.zeros((n, 1), dtype = np.float64)
        
        for clust in clusters:
            pi_k = clust['pi_k']
            mu_k = clust['mu_k']
            cov_k = clust['cov_k']
            resp = (pi_k * GMM.mvn(X, mu_k, cov_k)).astype(np.float64) #numerator of responsibility
            denom = np.sum(resp, dtype = np.float64) #denominator of responsibility        
            clust['resp'] = resp
            clust['denom'] = denom
        for clust in clusters:
            clust['resp'] /= clust['denom']
        
    def maximization(self, X, clusters):
        n, m = X.shape
        for clust in clusters:
            resp = clust['resp'] #responsibility
            cov_k = np.zeros((m, m))
            N_k = np.sum(resp, axis=0)
            pi_k = N_k / n #pi_k
            mu_k = np.sum(resp * X, axis=0) / N_k #mean
            
            for ii in range(n):
                z_mean = (X[ii] - mu_k).reshape(-1, 1)
                cov_k += resp[ii] * np.dot(z_mean, z_mean.T)
            cov_k /= N_k #covariance
            clust['pi_k'] = pi_k
            clust['mu_k'] = mu_k
            clust['cov_k'] = cov_k
    
    

    def fit(self, X, epochs, eps):
        '''Docstring
        
        Parameters
        --------------
           X: data matrix
           
        Attributes
        --------------
        
        
        '''
        self.X = X
        self.epochs = epochs
        self.eps = eps
        n, m = self.X.shape
        self.clusters = GMM.initialize_clusters(self.X, self.k)
        self.likelihoods = np.zeros((self.epochs, ))
        scores = np.zeros((n, self.k))
        
        for ii in range(self.epochs):
            self.expectation(self.X, self.clusters) #evaluates responsibility
            self.maximization(self.X, self.clusters) #evaluate parameters using responsibility
    
            self.likelihood = GMM.logLikelihood(self.X, self.clusters)
            self.likelihoods[ii] = self.likelihood
            print('Epoch: ', ii + 1, 'Likelihood: ', self.likelihood)
            #convergence test
            if len(self.likelihoods) == 1:
                pass
            else:
                if (np.abs(np.abs(self.likelihoods[ii]) - np.abs(self.likelihoods[ii-1])) <= eps) == True:
                    break
        for ij, clust in enumerate(self.clusters):
            scores[:, ij] = np.log(clust['resp']).reshape(-1)
        self.labels = np.array([np.argmax(scores[x]) for x in range(n)])
        return self
    
    def fit_predict(self, X, epochs, eps):
        self.X = X
        self.epochs = epochs
        self.eps = eps
        self.fit(self.X, self.epochs, self.eps)
        return self.labels
        
    def predict(self, X):
        '''Docstring
        
        '''
        return
        
        





        





        
        
        
        
