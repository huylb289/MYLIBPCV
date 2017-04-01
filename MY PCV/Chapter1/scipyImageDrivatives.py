from PIL import Image as IM
import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import os

# read image to array
dataFolder = '../../data'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

im = np.array(IM.open(os.path.join(dataFolder, 'boy_on_hill.jpg')).convert('L'))
# Sobel derivative filters
##This computes x and y derivatives and gradient magnitude using the Sobel filter. The
##second argument selects the x or y derivative, and the third stores the output.
imx = np.zeros(im.shape)
filters.sobel(im, 1, imx)

imy = np.zeros(im.shape)
filters.sobel(im, 0, imy)

magnitude = np.sqrt(imx**2 + imy**2)

plt.gray()
plt.imshow(imx)

plt.figure()
plt.gray()
plt.imshow(imy)

plt.figure()
plt.gray()
plt.imshow(magnitude)

plt.show()


##The filters.gaussian_filter() function we used for blurring earlier can also take extra
##arguments to compute Gaussian derivatives instead

sigma = 5 # standard deviation

imx = np.zeros(im.shape)
filters.gaussian_filter(im, (sigma, sigma), (0,1), imx)

imy = np.zeros(im.shape)
filters.gaussian_filter(im, (sigma, sigma), (1,0), imy)
