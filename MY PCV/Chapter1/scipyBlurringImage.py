##SciPy (http://scipy.org/) is an open-source package for mathematics that builds on
##NumPy and provides efficient routines for a number of operations, including numerical
##integration, optimization, statistics, signal processing, and most importantly for us,
##image processing.
from PIL import Image as IM
import numpy as np
from scipy.ndimage import filters # http://docs.scipy.org/doc/scipy/reference/ndimage.html
import matplotlib.pyplot as plt
import os



# read image to array
dataFolder = '../../data'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

im = np.array(IM.open(os.path.join(dataFolder, 'boy_on_hill.jpg')))
##Here the last parameter of gaussian_filter() is the standard deviation.
##im2 = filters.gaussian_filter(im, 5)

##To blur color images, simply apply Gaussian blurring to each color channel:
im2 = np.zeros(im.shape)
for i in range(3):
    im2[:,:,i] = filters.gaussian_filter(im[:,:,i], 10)

im2 = np.uint8(im2)

plt.figure()
plt.imshow(im2)
plt.show()


