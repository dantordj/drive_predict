from read_dataset import read_dataset
from imputation import *
from reduc_dim import *
from boosts import *
data = read_dataset('train.csv')
data2 = mean_imputation2(data)
#test = read_dataset('test.csv')
#test2 = mean_imputation(test)

#Y=LDA(data2)