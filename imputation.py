# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:33:53 2017

@author: bp
"""
import numpy as np
from sklearn.preprocessing import Imputer

def count(data):
    #compte la répartition des données en 0,1
    temp0 = 0
    temp1 = 0
    l,n=data.shape
    for i in range (l):
        if (data[i][n-1]==0):
            temp0 += 1
        else:
            temp1 += 1
    return (temp0,temp1)

def mean_computation(data):
    n0,n1 = count(data)
    l,n = data.shape
    means = [[0]*(n-1), [0]*(n-1)]
    l,n = data.shape
    for i in range (l):
        k=int(data[i][n-1])
        for j in range (n-1):
            means[k][j] += data[i][j]
    means[0] = [x / n0 for x in means[0]]
    means[1] = [x / n1 for x in means[1]]
    return means

def mean_test(data):
    n0,n1 = count(data)
    l,n = data.shape
    means1 = mean_computation(data)
    means2 = np.mean(data, axis=0)
    means3 = [0]*n
    for i in range(n-1):
        means3[i] = (n0*means1[0][i]+n1*means1[1][i])/(n0+n1)
        if(means2[i]-means3[i]!=0):
            print(means2[i]-means3[i])
    return (means2-means3)

def mean_imputation(data):
    l,n = data.shape
    means = mean_computation(data)
    for i in range(l):
        for j in range (n-1):
            if data[i][j] == -1:
                data[i][j] = means[int(data[i][n-1])][j]
    return data
