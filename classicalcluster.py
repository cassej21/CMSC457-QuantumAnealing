#Classical Analysis methods for clustering

import numpy as np
import cv2
#from scipy.cluster.vq import kmeans, whiten, vq
import matplotlib.pyplot as plt

IMAGE = "kirby.png" #32x32 px recommended max

image = cv2.imread("images/%s" %IMAGE, cv2.IMREAD_GRAYSCALE) #read image into array

H, W = image.shape[0], image.shape[1] # height and width of image
N = H * W # N = height * width
K = 3 # cluster counts
iterations = 100 #number of times to run kmeans for scipy
pixels = np.squeeze(np.reshape(image, (1, -1))) # flatten image into 1-D array
pixels = np.float32(pixels) #turn data type to float for opencv kmeans, comment out for scipy kmeans

# To run the scipy kmeans function, it is strongly encouraged to whiten the input
# This normalizes the features (rows) on a per observation (cols) basis
# Since the image is restructured to be 1 dimensional, this should not be necessary

#wpixels = whiten(pixels)
#codebook, distortion = kmeans(wpixels,K,iterations,.01) #scipy kmeans

#opencv implementation
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2) #Define stopping criteria, second value is max iteration count, third value is smallest change in cluster location
_, output, (centers) = cv2.kmeans(pixels,K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) #

centers = np.uint8(centers) #return pixel data to uchar
output = output.flatten()
segmented_image = centers[output.flatten()] #convert pixels to color of centroid
segmented_image = segmented_image.reshape(image.shape) #turn image back to original dimensions
# show the image
plt.imshow(segmented_image)
plt.show()
#plt.scatter(wpixels[:,0],wpixels[:, 1])
#plt.scatter(output[:,0],output[:, 1])
#plt.show()