from MYLIBPCV.tools import harris
from PIL import Image as IM
import numpy as np
import os

#http://en.wikipedia.org/wiki/Corner_detection.

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# read image and convert to graylevel
##imPIL = IM.open(os.path.join(dataFolder, 'empire.jpg'))
##im = np.array(imPIL.convert('L'))
##harrisim = harris.computeHarrisResponse(im)
##filteredCoords = harris.getHarrisPoints(harrisim, minDist=10, threshold=0.05)
##harris.plotHarrisPoints(im, filteredCoords)

# read left iamge and right image
imPIL = IM.open(os.path.join(dataFolder, 'crans_1_small.jpg'))
im1 = np.array(imPIL.convert('L'))

imPIL = IM.open(os.path.join(dataFolder, 'crans_2_small.jpg'))
im2 = np.array(imPIL.convert('L'))

threshold = 0.5
width = 5
im1Harris = harris.computeHarrisResponse(im1)
im1FilteredCoord = harris.getHarrisPoints(im1Harris, minDist=width+1, threshold=threshold)
im1Desc = harris.getDescriptors(im1, im1FilteredCoord, width=width)

im2Harris = harris.computeHarrisResponse(im2)
im2FilteredCoord = harris.getHarrisPoints(im2Harris, minDist=width+1,threshold=threshold)
im2Desc = harris.getDescriptors(im2, im2FilteredCoord, width=width)

print ('starting matching')
matches = harris.matchTwoSided(im1Desc,im2Desc)

harris.plotMatches(im1,im2,im1FilteredCoord,im2FilteredCoord,matches)
