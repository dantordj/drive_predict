from read_dataset import read_dataset
from imputation import *
from reduc_dim import *
from boosts import *
from predict_xgboost import *


print("preprocessing...")
data = read_dataset('train.csv')
data2_train = mean_imputation(data)

data_test = mean_imputation(read_dataset('test.csv'))
print("processing...")
predict_xgboost(data2_train, data_test)
#test = read_dataset('test.csv')
#test2 = mean_imputation(test)

#Y=LDA(data2)