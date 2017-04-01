##Morphology (or mathematical morphology) is a framework and a collection of image
##processing methods for measuring and analyzing basic shapes. Morphology is usually
##applied to binary images but can be used with grayscale also. A binary image is an
##image in which each pixel takes only two values, usually 0 and 1. Binary images are
##often the result of thresholding an image, for example with the intention of counting
##objects or measuring their size. A good summary of morphology and how it works is
##in http://en.wikipedia.org/wiki/Mathematical_morphology.

##Morphological operations are included in the scipy.ndimage module morphology .
##Counting and measurement functions for binary images are in the scipy.ndimage mod-
##ule measurements

from PIL import Image as IM
from scipy.ndimage import measurements, morphology
import numpy as np
import os
import matplotlib.pyplot as plt

# read image to array
dataFolder = '../../data'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# load image and threshold to make sure it is binary
im = np.array(IM.open(os.path.join(dataFolder, 'houses.png')).convert('L'))

##This loads the image and makes sure it is binary by thresholding. Multiplying by 1 con-
##verts the boolean array to a binary one.
im = 1*(im<128)

labels, nbr_objects = measurements.label(im)
print("Number of objects: {}".format(nbr_objects))

# morphology - opening to separate objects better
##The second argument of binary_opening() specifies the structuring element, an array
##that indicates what neighbors to use when centered around a pixel.

##In this case, we used 9 pixels (4 above, the pixel itself, and 4 below)
##in the y direction and 5 in the x direction

##The parameter iterations determines how many times to
##apply the operation. Try this and see how the number of objects changes.
im_open = morphology.binary_opening(im, np.ones((9,5)), iterations=2)

labels_open, nbr_objects_open = measurements.label(im_open)
print('Number of objects image open: {}'.format(nbr_objects_open))

plt.gray()
plt.imshow(im_open)
plt.show()
