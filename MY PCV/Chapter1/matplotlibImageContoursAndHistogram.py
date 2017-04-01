from PIL import Image as IM
import matplotlib.pyplot as plt
import numpy as np
import os

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# read image to array
##This needs grayscale images, because the contours need to be taken on a single
##value for every coordinate [x , y].
# set no use colors
imPIL = IM.open(os.path.join(dataFolder, 'empire.jpg'))
im = np.array(imPIL.convert('L'))

# create a new figure
plt.figure()
plt.gray()

# show contours with origin upper left corner
plt.contour(im, origin='image')
plt.axis('equal')
plt.axis('off')
plt.figure()

##As before, the PIL method convert() does conversion to grayscale.
##An image histogram is a plot showing the distribution of pixel values. A number of
##bins is specified for the span of values and each bin gets a count of how many pixels
##have values in the bin’s range. The visualization of the (graylevel) image histogram is
##done using the hist() function

##The method
##flatten() converts any array to a one-dimensional array with values taken row-wise
##The second argument specifies the number of bins to use.
plt.hist(im.flatten(), 10)
plt.show()
