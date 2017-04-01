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
im = np.array(imPIL.convert('L'))

im2 = 255 - im # invert image
im3 = (100.0/255) * im + 100 # clamp to interval 100...200
im4 = 255.0 * (im/255.0)**2 # squared

print(int(im.min()), int(im.max()))
print(int(im2.min()), int(im2.max()))
print(int(im3.min()), int(im3.max()))
print(int(im4.min()), int(im4.max()))

pil_im = IM.fromarray(im)
pil_im = IM.fromarray(np.uint8(im))
pil_im.show()

