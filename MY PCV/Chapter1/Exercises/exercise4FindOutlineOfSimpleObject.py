##Write a function that finds the outline of simple objects in images (for example, a
##square against white background) using image gradients.

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

sigma = 2
imBlur = np.zeros(im.shape)
filters.gaussian_filter(im, sigma, order=0, output=imBlur)

imGradient = np.zeros(im.shape)
filters.sobel(im, axis=-1, output=imGradient)

imBlurGradient = np.zeros(im.shape)
filters.sobel(imBlur, axis=-1, output=imBlurGradient)

plt.figure()
plt.gray()
plt.imshow(imGradient)

plt.figure()
plt.gray()
plt.imshow(imBlurGradient)

plt.show()
