##An alternative image normalization to histogram equalization
##is a quotient image. A
##quotient image is obtained by dividing the image with a
##blurred version I /(I ∗ Gσ ).
##Implement this and try it on some sample images.

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

quotientIm = im / (im * imBlur)

plt.figure()
plt.gray()
plt.imshow(im)

plt.figure()
plt.hist(im.flatten(),128)

plt.figure()
plt.gray()
plt.imshow(quotientIm)

plt.figure()
plt.hist(quotientIm.flatten(),128)

plt.show()
