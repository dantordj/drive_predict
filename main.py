from read_dataset import read_dataset
from imputation import *
from reduc_dim import *
data = read_dataset('train.csv')
data2 = mean_imputation(data)
#Y=LDA(data2)