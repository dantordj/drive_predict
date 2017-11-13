import numpy as np
import matplotlib.pyplot as plt 
import math

def PCA(data):
	N = len(data)
	scale(data)
	cov = np.dot(np.transpose(data),data) / N
	eig_values, eig_vectors = np.linalg.eig(cov)
	reduced = np.dot(np.transpose(eig_vectors), np.transpose(data))

	eig_values = np.sort(eig_values)
	#eig_values = np.flip(eig_values)
	sum_values = np.sum(eig_values)
	val_max = max(eig_values)
	for i in range(len(eig_values)):
		if eig_values[i] == val_max:
			print ("vector", eig_vectors[i])

	print(eig_values)
	list_for_gap = [e/sum_values for e in eig_values]
	X = range(len(eig_values))
	plt.plot(X, list_for_gap)
	plt.show()

	return reduced




def scale(data):
	print(data.shape)
	for j in range(len(data[:][0])):
		m = np.mean(data[:][j])
		v = np.std(data[:][j])
		data[:][j] = (data[:][j] - m) / math.sqrt(v)

r = PCA(r)

