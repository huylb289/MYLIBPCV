##An alternative corner detector to Harris is the FAST corner detector. There are a
##number of implementations, including a pure Python version available at http://
##www.edwardrosten.com/work/fast.html. Try this detector, play with the sensitivity
##threshold, and compare the corners with the ones from our Harris implementation.

from MYLIBPCV.tools import fast
from PIL import Image as IM
import numpy as np
import os
import matplotlib.pyplot as plt

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# read image and convert to graylevel
##imName = 'Univ1.jpg'
##imPath = os.path.join(dataFolder, imName)
##imPIL = IM.open(imPath)
##im = np.array(imPIL.convert('L'))
##fast.processImage(imPath, imName[:-3] + "fast", params='-l -t 100 -n 9')
##filteredCoords = fast.readFastPointsFromFile(imName[:-3] + 'fast')
##fast.plotFastPoints(im, filteredCoords)

# read left iamge and right image
imName1 = 'Univ1.jpg'
imPath1 = os.path.join(dataFolder, imName1)
imPIL1 = IM.open(imPath1)
im1 = np.array(imPIL1.convert('L'))
fast.processImage(imPath1, imName1[:-3] + "fast", params='-l -t 120 -n 9')
filteredCoords1 = fast.readFastPointsFromFile(imName1[:-3] + 'fast')
desc1 = fast.getDescriptors(im1, filteredCoords1)

imName2 = 'Univ2.jpg'
imPath2 = os.path.join(dataFolder, imName2)
imPIL2 = IM.open(imPath2)
im2 = np.array(imPIL2.convert('L'))
fast.processImage(imPath2, imName2[:-3] + "fast", params='-l -t 120 -n 9')
filteredCoords2 = fast.readFastPointsFromFile(imName2[:-3] + 'fast')
desc2 = fast.getDescriptors(im2, filteredCoords2)

print ('starting matching')
matches = fast.matchTwoSided(desc1,desc2, threshold=0.8)

plt.figure()
plt.gray()
fast.plotMatches(im1,im2,filteredCoords1,filteredCoords2,matches)
plt.show()
