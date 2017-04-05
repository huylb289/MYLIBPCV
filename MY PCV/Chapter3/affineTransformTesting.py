from scipy.ndimage import affine_transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as IM
import os
from MYLIBPCV.tools import homography

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# read image and convert to graylevel
imName = 'Univ1.jpg'
imPath = os.path.join(dataFolder, imName)
imPIL = IM.open(imPath)
im = np.array(imPIL.convert('L'))
H = np.array([[1.4,0.05,-100],[0.05,1.5,-100],[0,0,1]])
##H = homography.
imAffine = affine_transform(im, H[:2,:2], (H[0,2],H[1,2]))

plt.figure()
plt.gray()
plt.imshow(imAffine)
plt.show()
