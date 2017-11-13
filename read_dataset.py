import numpy as np

def read_dataset(filename):
    data = np.genfromtxt(filename, delimiter=",", names=True)
    return data
