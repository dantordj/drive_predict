# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:33:53 2017

@author: bp
"""
import numpy as np
import pandas as pd

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

<<<<<<< HEAD

def mean_computation(a):
    #prepare the training data (target column")
    #needs 
    data = a.drop(['id'], axis = 1)
    data.replace(-1, np.nan)
    if (data.shape[1]==58):
        means=data.groupby('target').mean()
    else :
        means=data.mean()
    data.replace(np.nan,-1)
=======
def mean_computation(data):
    means=data.groupby('ps_calc_20_bin'). mean()
>>>>>>> parent of 574631c... dataframe
    return means


def mean_imputation(data):
    data.astype(np.float16)
    means = mean_computation(data)
    means.astype(np.float16)
    l,n=data.shape
    print(l,n)
    if (n==59): 
       for i in range (l):
           if (i%10000==0):
               print(i)
           for j in means.columns: 
                if data.loc[i,j] == -1:
                    cat=int(data.loc[i,'target'])
                    data.loc[i,j] = means.loc[cat,j]
            
    else :
        for i in range (l):
            for j in means.columns:
                if data.loc[i,j] == -1:
                    data.loc[i,j] = means[j]
    return data
