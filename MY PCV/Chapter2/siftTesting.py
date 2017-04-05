from MYLIBPCV.tools import sift
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image as IM

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()


##imName = 'empire.jpg'
##imPath = os.path.join(dataFolder, imName)
##imPIL = IM.open(imPath)
##im = np.array(imPIL.convert('L'))
##sift.processImage(imPath, imName[:-4] + '.sift')
##l1, d1 = sift.readFeaturesFromFile(imName[:-4] + '.sift')
##plt.figure()
##plt.gray()
##sift.plotFeatures(im,l1, circle=True)
##plt.show()

# read image and convert to graylevel
imName1 = 'climbing_1_small.jpg'
imPath1 = os.path.join(dataFolder, imName1)
imPIL1 = IM.open(imPath1)
im1 = np.array(imPIL1)
sift.processImage(imPath1, imName1[:-4] + '.sift')
l1, d1 = sift.readFeaturesFromFile(imName1[:-4] + '.sift')

imName2 = 'climbing_2_small.jpg'
imPath2 = os.path.join(dataFolder, imName2)
imPIL2 = IM.open(imPath2)
im2 = np.array(imPIL2)
sift.processImage(imPath2, imName2[:-4] + '.sift')
l2, d2 = sift.readFeaturesFromFile(imName2[:-4] + '.sift')

matchscores = sift.matchTwoSided(d1,d2)

plt.figure()
plt.gray()
sift.plotMatches(im1,im2,l1,l2,matchscores, show_below=True)
plt.show()
