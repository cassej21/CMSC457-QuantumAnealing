# The QUBO formulation for clustering is specified in https://arxiv.org/pdf/1708.05753.pdf,
# with intensity / RGB difference as the distance metric.

from dwave_qbsolv import *
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE = "kirby.jpg" # file we are analyzing ... should only be about 32 x 32 sadly

image = cv2.imread("images/%s" % IMAGE, cv2.IMREAD_GRAYSCALE) # read image into numpy array

H, W = image.shape[0], image.shape[1] # height and width of image
N = H * W # N = height * width
K = 2 # cluster counts
n = 6 # hardware precision of annealer coupling / bias
LAMBDA = (2 ** (n-1) - 1) / (2 * (K - 2 if K > 2 else K)) # calculate lambda cluster constraint
# SCALE = (2 ** n - 2) / ((N - K) * (K - 2 if K > 2 else K)) # calculate required magnitude scaling
SCALE = 0.25

print("Scaling all loss to %f" % SCALE)

pixels = np.squeeze(np.reshape(image, (1, -1))) # flatten image into 1-D array
diffs = np.zeros((N, N)) # store pairwise differences
sums = np.zeros(N) # store sum of all pairwise differences relative to i-th pixel

for idx, pix in enumerate(pixels): # for all pixels in the image...
    diffs[idx] = (pixels - pix) * SCALE / 255 # assign pairwise difference for all pixels, scaled accordingly
    sums[idx] = np.sum(diffs[idx]) # assign sum of all pairwise differences relative to i-th pixel

h = {} # intrinsic weight matrix
J = {} # coupling weight matrix

# assign general pairwise clustering weights - both biases and couplings
for k in range(K): # for every cluster we have...
    for i in range(N): # for every pixel in the image...
        for j in range(N): # and every pixel it might be paired with...
            # give each binary variable an unique number
            var_i = 10 * i + k # s_ik = x_(10i+k)
            var_j = 10 * j + k # s_jk = x_(10j+k)
            if i == j: # intrinsic weights (bias) indexing - for every cluster assignment (s_ik for all k in K)
                h[var_i] = sums[i] + (2 * LAMBDA * (K - 2 if K > 2 else K)) # assign weight
            else:
                J[(var_i, var_j)] = 0.5 * diffs[i][j] # the weighting for a cluster assignment for two vars is half their "distance"

# assign pairwise clustering variables per pixel
for i in range(N):
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

clusters = list(response.samples())[0] # store clustering assignments
clustered = np.zeros(N) # initialize empty array

for key in clusters.keys():
    key_str = str(key)
    if key == 0: key_str = "00"
    if key == 1: key_str = "01"
    key_pix, key_cluster, key_status = int(key_str[:-1]), int(key_str[-1:]), clusters[key]
    if key_status == 1:
        clustered[key_pix] = 255

clustered = np.reshape(clustered, (H, W))

fig, axes = plt.subplots(1, 2)

axes[0].imshow(image)
axes[1].imshow(clustered)
plt.show()