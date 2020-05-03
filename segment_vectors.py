# The QUBO formulation for clustering is specified in https://arxiv.org/pdf/1708.05753.pdf,
# with intensity / RGB difference as the distance metric.

from dwave_qbsolv import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import path

DATA = path.join("iris", "bezdekIris.data") # path to data file

vectors = None # Collection of D n-dimensional vectors. Must be of shape (D, n)

# Parse data file.
fi = open(path.join("datasets", DATA))
lines = [line.strip().split(',') for line in fi.readlines() if line.strip() != '']

values = [[float(entry) for entry in line[:-1]] for line in lines]
name_numbers = { 'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2 }
labels = [name_numbers[line[-1]] for line in lines]

values = np.array(values)

# "Load" parsed dataset.
vectors = values

if vectors is None or len(vectors.shape) != 2:
    print("Specify a valid dataset first.")
    exit()

N, C = vectors.shape # Number of vectors and their dimensionality (C for "channels").
K = 3 # cluster counts
n = 7 # hardware precision of annealer coupling / bias
LAMBDA = (2 ** (n-1) - 1) / (2 * (K - 2 if K > 2 else K)) # calculate lambda cluster constraint
SCALE = 1

SNR = np.linalg.norm(np.mean(np.mean(vectors, axis=0), axis=0)) / np.linalg.norm(np.std(np.std(vectors, axis=0), axis=0))

print("Found input SNR to be %f." % SNR)
print("Scaling all loss to %f" % SCALE)

diffs = np.zeros((N, N)) # store pairwise differences
sums = np.zeros(N) # store sum of all pairwise differences relative to i-th pixel

print("Calculating pairwise differences...")

for idx, vec in tqdm(enumerate(vectors)): # for all pixels in the image...
    diffs[idx] = np.linalg.norm((vectors - vec), axis=1) * SCALE # assign pairwise difference for all pixels, scaled accordingly
    sums[idx] = np.sum(diffs[idx]) # assign sum of all pairwise differences relative to i-th pixel

h = {} # intrinsic weight matrix
J = {} # coupling weight matrix

print("Assigning pairwise clustering weights...")

# assign general pairwise clustering weights - both biases and couplings
for k in tqdm(range(K)): # for every cluster we have...
    for i in range(N): # for every pixel in the image...
        for j in range(N): # and every pixel it might be paired with...
            # give each binary variable an unique number
            var_i = 10 * i + k # s_ik = x_(10i+k)
            var_j = 10 * j + k # s_jk = x_(10j+k)
            if i == j: # intrinsic weights (bias) indexing - for every cluster assignment (s_ik for all k in K)
                h[var_i] = sums[i] + (2 * LAMBDA * (K - 2 if K > 2 else K)) # assign weight
            else:
                J[(var_i, var_j)] = 0.5 * diffs[i][j] # the weighting for a cluster assignment for two vars is half their "distance"

print("Assigning pairwise cluster possibility weights...")

# assign pairwise clustering variables per pixel
for i in tqdm(range(N)):
    for a in range(K):
        for b in range(K):
            if a != b:
                var_a = 10 * i + a
                var_b = 10 * i + b
                J[(var_a, var_b)] = LAMBDA

# TODO: Guarantee that for all s_ik for each i, only one is "true"

print("%d couplings defined between %d variables." % (len(J.keys()), N * K))

response = QBSolv().sample_ising(h, J, verbosity=2) # sample / solve Ising problem

print(response)

clusters = list(response.samples()) # store clustering assignments
clustered = np.zeros(N)

accuracies = [] # record accuracies of every given solution
assignments = [] # record corresponding assignments

for i, clustering in enumerate(clusters): # for every low-energy solution...
    clustered = np.zeros(N) # initialize empty array

    # mark cluster assignments for visualization
    for key in clustering.keys():
        # extract pixel number and cluster assignments (var. number format is {pixel}{cluster})
        key_str = str(key)
        # since pixel #0 cluster assignments will break this format (only {cluster}), fix manually
        if len(key_str) == 1:
            key_str = "0%s" % key_str
        # extract all parameters from single number!
        key_pix, key_cluster, key_status = int(key_str[:-1]), int(key_str[-1:]), clustering[key]
        # if spin / state is +1, fill in cluster indicated by variable
        if key_status == 1:
            clustered[key_pix] = key_cluster

    # Count matches
    matches = np.sum(np.where(clustered == labels, 1, 0))
    percent = matches / N

    # Record accuracy
    accuracies.append(percent)

    # Record assignment
    assignments.append(clustered)

accuracies = np.array(accuracies)

print(accuracies)

print("Highest accuracy: %f percent" % (np.max(accuracies) * 100))
print("Associated clustering: %s" % str(assignments[np.argmax(accuracies)]))