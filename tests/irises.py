import sys
from os import path
sys.path.append(path.realpath('../'))
from segment_vectors import ising_k
import numpy as np
import matplotlib.pyplot as plt

DATA = path.join("iris", "bezdekIris.data") # path to data file

# Parse data file.
fi = open(path.realpath(path.join("../datasets", DATA)))
lines = [line.strip().split(',') for line in fi.readlines() if line.strip() != '']

values = [[float(entry) for entry in line[:-1]] for line in lines]
name_numbers = { 'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2 }
labels = [name_numbers[line[-1]] for line in lines]

values = np.array(values)

results = ising_k(values, labels)