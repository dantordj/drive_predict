from read_dataset import read_dataset
from imputation import *
from reduc_dim import *
data = read_dataset('train.csv')
print(type(data))
print(len(data), " rows")
print(len(data[0]), " columns")
print(data[0])
data2 = mean_imputation(data)
Y=LDA(data2)