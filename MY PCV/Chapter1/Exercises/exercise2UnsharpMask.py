##2.Implement an unsharp masking operation (http://en.wikipedia.org/wiki/Unsharp_
##masking) by blurring an image and then subtracting the blurred version from the
##original. This gives a sharpening effect to the image. Try this on both color and
##grayscale images.


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
imGray = np.array(imPIL.convert('L'))
imColor = np.array(imPIL)

# Gray image
imGrayBlur = np.zeros(imGray.shape)
sigma = 3

##order : {0, 1, 2, 3} or sequence from same set, optional
##The order of the filter along each axis is given as a sequence of integers,
##or as a single number. An order of 0 corresponds to convolution
##with a Gaussian kernel. An order of 1, 2, or 3 corresponds to convolution
##with the first, second or third derivatives of a Gaussian.
##Higher order derivatives are not implemented
filters.gaussian_filter(imGray, sigma, order=0, output=imGrayBlur)
imGrayUnSharp = imGrayBlur - imGray

# Color image
imColorBlur = np.zeros(imColor.shape)

for channel in range(3):
    filters.gaussian_filter(imColor[:,:,channel], sigma, order=0, output=imColorBlur[:,:,channel])
imColorUnSharp = imColorBlur - imColor

##plt.figure()
##plt.gray()
##plt.axis('equal')
##plt.axis('off')
##plt.imshow(imGrayUnSharp)
##
##plt.figure()
##plt.axis('equal')
##plt.axis('off')
##plt.imshow(imColorUnSharp)
##
##plt.show()

##imsave('imColorUnSharp.jpg',imColorUnSharp)
##imsave('imGrayUnSharp.jpg',imGrayUnSharp)

pilIm = IM.fromarray(np.uint8(imGrayUnSharp))
pilIm.show()

pilIm = IM.fromarray(np.uint8(imColorUnSharp))
pilIm.show()

