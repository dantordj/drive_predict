# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:09:44 2017

@author: bp
"""

import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

def train_xboost(data):
    X = data.drop(['target','id'], axis = 1)
    y = data['target']
    print(X)
    # fit model no training data
    #model = xgb.XGBClassifier()
    #model.fit(X, y)
    # feature importance
    print(model.feature_importances_)
    # plot
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.show()
    return model

def predict_xboost(data,model):
    pred = model.predict(data)
    return pred