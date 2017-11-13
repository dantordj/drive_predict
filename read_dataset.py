import numpy as np

def read_dataset(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    return data
