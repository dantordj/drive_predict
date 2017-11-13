import numpy as np
import csv as csv

def read_dataset(filename):
    data = []
    with open(filename, newline='') as my_file:
        reader = csv.reader(my_file)
        header_line = True
        for row in reader:
            if header_line == True:
                labels = row
                header_line = False
            else:
                data += [row]

    return data, labels
