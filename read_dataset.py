import numpy as np
import pandas as pd

def read_dataset(filename):
    data = pd.read_csv(filename)
    return data
