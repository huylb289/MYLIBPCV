from PIL import Image as IM
import matplotlib.pyplot as plt
import numpy as np
import os
from MYLIBPCV.tools import imtools

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# get list image
imlist = imtools.getImlist(dataFolder)
averageIm = imtools.computeAverage(imlist)


# create a new figure
plt.figure()
##plt.gray()

plt.imshow(averageIm)
plt.show()
