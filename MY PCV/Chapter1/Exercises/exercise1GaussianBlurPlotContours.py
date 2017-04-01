##1.Take an image and apply Gaussian blur like in Figure 1-9. Plot the image contours
##for increasing values of σ . What happens? Can you explain why?

##Image loose the weak edges after using Gaussian bluring, and contour

from PIL import Image as IM
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import filters

# read image to array
dataFolder = '../../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# read image, convert to gray channel, and turn it to array
imPIL = IM.open(os.path.join(dataFolder, 'empire.jpg'))
im = np.array(imPIL.convert('L'))

sigma = 5
imGauBlur1 = np.zeros(im.shape)
filters.gaussian_filter(im, sigma, order=1, output=imGauBlur1)

imGauBlur2 = np.zeros(im.shape)
filters.gaussian_filter(im, sigma, order=0, output=imGauBlur2)


magnitudeIm = np.sqrt(imGauBlur1**2 + imGauBlur2**2)

# create new figure and plot contour
plt.figure()
plt.gray()
plt.contour(im, origin='image')
plt.axis('equal')
plt.axis('off')

plt.figure()
plt.gray()
plt.contour(imGauBlur1, origin='image')
plt.axis('equal')
plt.axis('off')

plt.figure()
plt.gray()
plt.contour(imGauBlur2, origin='image')
plt.axis('equal')
plt.axis('off')

plt.figure()
plt.gray()
plt.contour(magnitudeIm, origin='image')
plt.axis('equal')
plt.axis('off')

plt.show()
