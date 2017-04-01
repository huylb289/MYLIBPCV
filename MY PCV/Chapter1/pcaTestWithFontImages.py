from PIL import Image as IM
import numpy as np
from MYLIBPCV.tools import imtools, pca
import os
import matplotlib.pyplot as plt

# read image to array
dataFolder = '../../data/a_thumbs'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

imList = imtools.getImlist(dataFolder)
im = np.array(IM.open(imList[0])) # open one image to get the size
m, n = im.shape[0:2] # get the size of the images(0:2 to not get the color channel)
imnbr = len(imList) # get the number of images

# create matrix to store all flattened images
immatrix = np.array([np.array(IM.open(im)).flatten() \
                     for im in imList], 'f')

# perform PCA
V,S,immean = pca.pca(immatrix)

# show some images (mean and 7 first modes)
plt.figure()
plt.gray()
plt.subplot(2,4,1)
plt.imshow(immean.reshape(m,n))
for i in range(7):
    plt.subplot(2,4,i+2)
    plt.imshow(V[i].reshape(m,n))

plt.show()

import pickle

# save mean and principal components into pickle module
##http://docs.python.org/library/pickle.html
f = open('font_pca_models.pkl', 'wb')
pickle.dump(immean,f)
pickle.dump(V,f)
f.close()

# load mean and principal components
f = open('font_pca_models.pkl', 'rb')
immean = pickle.load(f)
V = pickle.load(f)
f.close()

# python2
### open file and save
##with open('font_pca_modes.pkl', 'wb') as f:
##pickle.dump(immean,f)
##pickle.dump(V,f)

### open file and load
##with open('font_pca_modes.pkl', 'rb') as f:
##immean = pickle.load(f)
##V = pickle.load(f)

# Alternative way to save variable into files
##http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
##The last parameter indicates that integer format should be used.
np.savetxt('immean.txt', immean, '%i')

#load txt files
immean = np.loadtxt('immean.txt')
