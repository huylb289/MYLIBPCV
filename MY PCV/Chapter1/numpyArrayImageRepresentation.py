##NumPy (http://www.scipy.org/NumPy/) is a package popularly used for scientific comput-
##ing with Python. NumPy contains a number of useful concepts such as array objects (for
##representing vectors, matrices, images and much more) and linear algebra functions.

from PIL import Image as IM
import matplotlib.pyplot as plt
import numpy as np
import os

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# read image
imPIL = IM.open(os.path.join(dataFolder, 'empire.jpg'))
im = np.array(imPIL)
##The first tuple on each line is the shape of the image array (rows, columns, color
##channels)
print(im.shape, im.dtype)

i = 100
j = 100
k = 0

value = im[i,j,k] 
im[i,:] = im[j,:] # set the values of row i with values from row j
im[:,i] = 100 # set all values in column i to 100
im[:100,:50].sum() # the sum of the values of the first 100 rows and 50 columns
im[50:100,50:100] # rows 50-100, columns 50-100 (100th not included)
im[i].mean() # average of row i
im[:,-1] # last column
im[-2,:] # (or im[-2])

im = np.array(imPIL.convert('L'))
print(im.shape, im.dtype)

