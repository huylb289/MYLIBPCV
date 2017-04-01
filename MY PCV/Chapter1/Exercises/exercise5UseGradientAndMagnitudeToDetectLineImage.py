##Use gradient direction and magnitude to detect lines in an image. Estimate the
##extent of the lines and their parameters. Plot the lines overlaid on the image.

from PIL import Image as IM
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import filters
from scipy.misc import imsave

# read image to array
dataFolder = '../../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# read image, convert to gray channel, and turn it to array
imPIL = IM.open(os.path.join(dataFolder, 'empire.jpg'))
im = np.array(imPIL.convert('L'))

# Simple Edge Detection Using Gradients Direction and Magnitude
# 1. Compute gradient vector X direction, and Y direction
# 2. Compute gradient magnitude
# 3. Eliminate all pixels that lesser than threshold

imGradX = np.zeros(im.shape)
filters.sobel(im, axis=1, output=imGradX)

imGradY = np.zeros(im.shape)
filters.sobel(im, axis=0, output=imGradY)

threshold = 0.3
imMagnitude = np.sqrt(imGradX**2 + imGradY**2)

imMagnitudeNormed = imMagnitude / imMagnitude.max()
imFinalEdge = (imMagnitudeNormed > threshold) * 1

plt.gray()
plt.axis('equal')
plt.axis('off')
plt.imshow(imFinalEdge)

plt.show()
