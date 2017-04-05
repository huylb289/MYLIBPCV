from MYLIBPCV.tools import warp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as IM
from MYLIBPCV.tools import homography
from scipy.ndimage import affine_transform
import os
from scipy.ndimage import filters

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()


# blur image to get a better result
# example of affine warp of im1 onto im2
im1Name = 'beatles.jpg'
im2Name = 'billboard_for_rent.jpg'
imPath = os.path.join(dataFolder, im1Name)
imPIL = IM.open(imPath)
im1 = np.array(imPIL.convert('L'))
im1Result = np.zeros(im1.shape)
filters.gaussian_filter(im1, sigma=10, order=0, output=im1Result)


imPath = os.path.join(dataFolder, im2Name)
imPIL = IM.open(imPath)
im2 = np.array(imPIL.convert('L'))
im2Result = np.zeros(im2.shape)
filters.gaussian_filter(im2, sigma=5, order=0, output=im2Result)

# set to points
##tp = np.array([[264,538,540,264],[40,36,605,605],[1,1,1,1]])
tp = np.array([[675,826,826,677],[55,52,281,277],[1,1,1,1]])
##im3 = warp.imageInImage(im1,im2,tp)
im3 = warp.imageInImage(im1Result,im2,tp)

# set to points
##tp = np.array([[264,538,540,264],[40,36,605,605],[1,1,1,1]])
tp = np.array([[675,826,826,677],[55,52,281,277],[1,1,1,1]])
##im3 = warp.imageInImage(im1,im2,tp)
im3 = warp.imageInImage(im1Result,im2,tp)

plt.figure()
plt.gray()
plt.imshow(im3)
plt.axis('equal')
plt.axis('off')
plt.show()
