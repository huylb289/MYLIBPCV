import numpy as np
from MYLIBPCV.tools import imtools
from PIL import Image as IM
import os
import matplotlib.pyplot as plt

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# read image
imPIL = IM.open(os.path.join(dataFolder, 'AquaTermi_lowcontrast.JPG'))
im = np.array(imPIL.convert('L'))
im2, cdf = imtools.histeq(im)


##MxN – values to be mapped (float or int)
##MxNx3 – RGB (float or uint8)
##MxNx4 – RGBA (float or uint8)

print(im.shape, im.dtype)
print(im2.shape, im2.dtype)

plt.gray()
plt.axis('off')
plt.imshow(im)

plt.figure()
plt.axis('off')
plt.imshow(im2)


plt.show()
