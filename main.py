from read_dataset import read_dataset
from imputation import *
from reduc_dim import *
from predict_xgboost import *
from clustering_missing import *


print("preprocessing...")
data_train = read_dataset('train.csv')
#data2_train = mean_imputation(data_train)


data_train_modified = cluster_missing(data_train)

data_test_ini = read_dataset('test.csv')
data_test_modified = cluster_missing(data_test_ini)
#data_test = mean_imputation(read_dataset('test.csv'))
print("processing...")
s1,y1=test(data_train,data_test_ini)
s2,y2=test(data_modified,data_modified)
predict_xgboost(data_train_modified, data_test_modified)
