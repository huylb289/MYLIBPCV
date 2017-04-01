import numpy as np
from scipy.misc import imsave
import scipy
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

lena = scipy.misc.ascent()

imsave('test.jpg',im)
imsave('lena.jpg',lena)
