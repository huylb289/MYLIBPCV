from scipy.cluster import vq
from scipy.misc import imresize
from PIL import Image as IM
import numpy as np
import os
import matplotlib.pyplot as plt

dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

imPath = os.path.join(dataFolder, 'empire.jpg')

steps = 50 # image is divided in steps*steps region
im = np.array(IM.open(imPath))

dx = im.shape[0] / steps
dy = im.shape[1] / steps

# compute color features for each region
features = []
for x in range(steps):
    for y in range(steps):
        slicex1 = int(x*dx)
        slicex2 = int((x+1)*dx)
        slicey1 = int(y*dy)
        slicey2 = int((y+1)*dy)
        R = np.mean(im[slicex1:slicex2,slicey1:slicey2,0])
        G = np.mean(im[slicex1:slicex2,slicey1:slicey2,1])
        B = np.mean(im[slicex1:slicex2,slicey1:slicey2,2])
        features.append([R,G,B])

features = np.array(features, 'f') # make into array

# cluster
centroids, variance = vq.kmeans(features, 3)
code, distance = vq.vq(features, centroids)
"""
The input to k-means is an array with steps*steps rows, each containing the R, G, and
B mean values. To visualize the result, we use SciPy ’s imresize() function to show the
steps*steps image at the original image coordinates. The parameter interp specifies what
type of interpolation to use; here we use nearest neighbor so we don’t introduce new
pixel values at the transitions between classes.
"""

# create image with cluster labels
codeim = code.reshape(steps, steps)
codeim = imresize(codeim, im.shape[:2], interp='nearest')

plt.figure()
plt.imshow(codeim)
plt.show()
