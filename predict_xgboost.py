# This script shows you how to make a submission using a few
# useful Python libraries.
# It gets a public leaderboard score of 0.76077.
# Maybe you can tweak it and do better...?

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold
from gini import *

    
def predict_xgboost(train_df, test_df):

	# We'll impute missing values using the median for numeric columns and the most
	# common value for string columns.
	# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948
    feature_columns_to_use = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']
    #feature_columns_to_use = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15']
        
        
    # Prepare the inputs for the model
    print(train_df.shape[0])
    train_X = train_df[feature_columns_to_use]
    test_X = test_df[feature_columns_to_use]
    train_y = train_df['target']
        
    # You can experiment with many other options here, using the same .fit() and .predict()
    # methods; see http://scikit-learn.org
    # This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
    #gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05, objective='multi:softprob').fit(train_X, train_y)
        
    gbm = xgb.XGBClassifier().fit(train_X, train_y)
        
    predictions = gbm.predict_proba(test_X)
    print(predictions)
        	# Kaggle needs the submission to have a certain format;
        	# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
        	# for an example of what it's supposed to look like
    return predictions

        
def predict_xgboost_k_fold(data,k,test_v):       
    
    kf=KFold(n_splits=k)
    results=[]
        
	# We'll impute missing values using the median for numeric columns and the most
	# common value for string columns.
	# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948
    feature_columns_to_use = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']
    #feature_columns_to_use = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15']
    
    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
    #gbm = xgb.XGBClassifier()
    # Prepare the inputs for the model
    for train, test in kf.split(data):
        
        train_data = data.loc[train]
        test_data = data.loc[test]
        train_X = train_data[feature_columns_to_use]
        test_X = test_data[feature_columns_to_use]
        train_y = train_data['target']
        gbm=gbm.fit(train_X, train_y)
        predictions = gbm.predict_proba(test_X)
        results.append(gini_xgb(predictions, test_data))    
        	# Kaggle needs the submission to have a certain format;
        	# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
        	# for an example of what it's supposed to look like
    test_v=test_v[feature_columns_to_use]
    pred=gbm.predict_proba(test_v)
    return np.mean(results),pred

def submit(predictions, test_df):
    submission = pd.DataFrame({ 'id': [int(a) for a in test_df['id']],'target': predictions[:,1] })
    print(submission)
    submission.to_csv("submission.csv", index=False)