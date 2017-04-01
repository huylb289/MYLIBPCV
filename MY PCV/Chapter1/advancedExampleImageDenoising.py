##from numpy import *
##from numpy import random
##from scipy.ndimage import filters
##from MYLIBPCV.tools import rof
##
### create synthetic image with noise
##im = zeros((500,500))
##im[100:400,100:400] = 128
##im[200:300,200:300] = 255
##im = im + 30*random.standard_normal((500,500))
##U,T = rof.denoise(im,im)
##G = filters.gaussian_filter(im,10)
##
### save the result
##from scipy.misc import imsave
##imsave('synth_rof.pdf',U)
##imsave('synth_gaussian.pdf',G)

from PIL import Image as IM
from pylab import *
from MYLIBPCV.tools import rof
import os

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# read image to array
imPIL = IM.open(os.path.join(dataFolder, 'empire.jpg')).convert('L')
im = array(imPIL)

U,T = rof.denoise(im,im)
figure()
gray()
imshow(U)
axis('equal')
axis('off')
show()
