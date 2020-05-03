# The QUBO formulation for binary clustering is specified in https://arxiv.org/pdf/1708.05753.pdf,
# with intensity / RGB difference as the distance metric.

from dwave_qbsolv import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

IMAGE = "jon.jpg" # file we are analyzing ... should only be about 32 x 32 sadly

image = cv2.imread("images/%s" % IMAGE, cv2.IMREAD_GRAYSCALE) # read image into numpy array

H, W = image.shape[0], image.shape[1] # height and width of image
N = H * W # N = height * width
K = 2 # cluster counts
SCALE = 900

SNR = np.linalg.norm(np.mean(np.mean(image, axis=0), axis=0)) / np.linalg.norm(np.std(np.std(image, axis=0), axis=0))

print("Found input SNR to be %f." % SNR)
print("Scaling all loss to %f" % SCALE)

pixels = np.squeeze(np.reshape(image, (1, -1))) # flatten image into 1-D array
diffs = np.zeros((N, N)) # store pairwise differences
sums = np.zeros(N) # store sum of all pairwise differences relative to i-th pixel

print("Calculating pairwise differences...")

for idx, pix in tqdm(enumerate(pixels)): # for all pixels in the image...
    diffs[idx] = (pixels - pix) * SCALE / 255 # assign pairwise difference for all pixels, scaled accordingly
    sums[idx] = np.sum(diffs[idx]) # assign sum of all pairwise differences relative to i-th pixel

h = {}
J = {}

print("Assigning pairwise cluster weights...")

for i in tqdm(range(N)):
    for j in range(N):
        J[(i, j)] = 0.5 * diffs[i][j]
    

response = QBSolv().sample_ising(h, J, verbosity=2) # sample / solve Ising problem

clusters = list(response.samples()) # store clustering assignments
fig, axes = plt.subplots(1, len(clusters) + 1) # create plot to map all resulting samples
intensities = [(255.0 / (K-1)) * i for i in range(K)] # prepare intensities for displaying k clusters (from 0 to 255)


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
        key_pix, key_status = int(key_str), clustering[key]
        # if spin / state is +1, fill in cluster indicated by variable
        if key_status == 1:
            clustered[key_pix] = intensities[0]
        else:
            clustered[key_pix] = intensities[1]

    clustered = np.reshape(clustered, (H, W))
    axes[i + 1].imshow(clustered)

axes[0].imshow(image)
plt.show()