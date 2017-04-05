from MYLIBPCV.tools import harris
from PIL import Image as IM
import numpy as np
import os
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import cv2
from MYLIBPCV.tools import rof

#http://en.wikipedia.org/wiki/Corner_detection.

##Incrementally apply stronger blur (or ROF de-noising) to an image and extract
##Harris corners. What happens?

## Harris corners are not affected by gaussian
## Denoise using ROF affect Harris Corner Detection

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# read image
imPIL = IM.open(os.path.join(dataFolder, 'crans_1_small.jpg'))
im = np.array(imPIL.convert('L'))

imBlur =np.zeros(im.shape)
sigma = 10
filters.gaussian_filter(im, sigma, order=0, output=imBlur)

# read image and convert to graylevel
##harrisim = harris.computeHarrisResponse(imBlur)
##filteredCoords = harris.getHarrisPoints(harrisim, minDist=10, threshold=0.05)
##
##plt.figure()
##plt.gray()
##harris.plotHarrisPoints(imBlur, filteredCoords)
##plt.show()

# Denoise using rof
U,T = rof.denoise(im,im)
# read image and convert to graylevel
harrisim = harris.computeHarrisResponse(U)
filteredCoords = harris.getHarrisPoints(U, minDist=10, threshold=0.05)

plt.figure()
plt.gray()
harris.plotHarrisPoints(U, filteredCoords)
plt.show()
