# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:33:53 2017

@author: bp
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

def get_values(data):
    values=np.array([0]*data.shape[1])
    for i in range (data.shape[1]):
        values[i]=len(np.unique(data[:,i]))
    return values

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
    means=data.groupby('ps_calc_20_bin'). mean()
    return means


def mean_imputation(data):
    l,n = data.shape
    means = mean_computation(data).values
    data=data.values
    for i in range (l):
        for j in range(n-1):
            if data[i][j] == -1:
                data[i][j] = means[int(data[i][n-1])][j]
    return pd.DataFrame(data)


