#%%
import numpy as np
import seaborn as sns
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import datasets
import pandas as pd
import sys
from os import path
sys.path.append(path.realpath('../'))
import numpy as np
import matplotlib.pyplot as plt

DATA = path.join("iris", "bezdekIris.data") # path to data file

# Parse data file.
fi = open(path.realpath(path.join("datasets", DATA)))
lines = [line.strip().split(',') for line in fi.readlines() if line.strip() != '']

values = [[float(entry) for entry in line[:-1]] for line in lines]
name_numbers = { 'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2 }
labels = [name_numbers[line[-1]] for line in lines]

values = np.array(values)
print(values)

print(values[1])


t_Vals = values.transpose()





#A,B,C,D= data.shape[0],data.shape[1],data.shape[2],data.shape[3]




#x=np.random.randint(0,np.max(A))







# %%
