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

plt.imshow(im)

print('Please click 3 points')
x = plt.ginput(3)
print('You clicked {}'.format(x))
plt.show()
