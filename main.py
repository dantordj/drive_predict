from read_dataset import read_dataset

data = read_dataset('train.csv')
print(type(data))
print(len(data), " rows")
print(len(data[0]), " columns")
print(data[0])