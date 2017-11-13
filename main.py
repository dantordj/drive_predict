from read_dataset import read_dataset

data, labels = read_dataset('train.csv')
print(labels)
print(type(data))
print(data)
##print(len(data), " rows")
##print(len(data[0]), " columns")
##for i in range(100):
##   print(data[i])