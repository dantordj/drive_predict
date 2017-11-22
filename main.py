from read_dataset import read_dataset
from imputation import *
from reduc_dim import *
from predict_xgboost import *


print("preprocessing...")
data = read_dataset('train.csv')
data2_train = mean_imputation(data)

#data_test = mean_imputation(read_dataset('test.csv'))
print("processing...")
k_fold(2,data)
k_fold(2,data2_train)
#test = read_dataset('test.csv')
#test2 = mean_imputation(test)

#Y=LDA(data2)