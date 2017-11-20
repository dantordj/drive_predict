from read_dataset import read_dataset
from imputation import *
from reduc_dim import *
from boosts import *
data = read_dataset('train.csv')
<<<<<<< HEAD
test = read_dataset('test.csv')
#data2 = mean_imputation(data)
#test2 = mean_imputation(test)
=======

print(data.shape)


data2 = mean_imputation(data)
>>>>>>> parent of 574631c... dataframe
#Y=LDA(data2)