##Experiment with successive morphological operations on a thresholded image of
##your choice. When you have found some settings that produce good results, try the
##function center_of_mass in morphology to find the center coordinates of each object
##and plot them in the image.

from PIL import Image as IM
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import filters
from scipy.misc import imsave
from scipy.ndimage import measurements, morphology

# read image to array
dataFolder = '../../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# read image, convert to gray channel, and turn it to array
imPIL = IM.open(os.path.join(dataFolder, 'houses.png'))
im = np.array(imPIL.convert('L'))

im = 1*(im<128)

labels, nbr_objects = measurements.label(im)
print("Number of objects: {}".format(nbr_objects))

# Calculate in single object
centers = measurements.center_of_mass(im)

# Calculate multiple objects
##centers = measurements.center_of_mass(im, nbr_objects, [1,2])
print(centers)

plt.gray()
plt.axis('equal')
plt.axis('off')
plt.imshow(im)

plt.plot(centers[0], centers[1], 'r*')
    
plt.show()


# Using image normal and detect object 

# Simple Edge Detection Using Gradients Direction and Magnitude
# 1. Compute gradient vector X direction, and Y direction
# 2. Compute gradient magnitude
# 3. Eliminate all pixels that lesser than threshold

##imGradX = np.zeros(im.shape)
##filters.sobel(im, axis=1, output=imGradX)
##
##imGradY = np.zeros(im.shape)
##filters.sobel(im, axis=0, output=imGradY)
##
##threshold = 100
##imMagnitude = np.sqrt(imGradX**2 + imGradY**2)
##
##imFinalEdge = (imMagnitude > threshold) * 1
##
##
##labels, nbr_objects = measurements.label(im)
##print("Number of objects: {}".format(nbr_objects))
##
##centers = measurements.center_of_mass(im)
##
##plt.gray()
##plt.axis('equal')
##plt.axis('off')
##plt.imshow(imFinalEdge)
##    
##plt.show()
