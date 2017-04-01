##When working with mathematics and plotting graphs or drawing points, lines, and
##curves on images, Matplotlib is a good graphics library with much more powerful
##features than the plotting available in PIL

from PIL import Image as IM
import matplotlib.pyplot as plt
import numpy as np
import os

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# read image to array
imPIL = IM.open(os.path.join(dataFolder, 'empire.jpg'))
im = np.array(imPIL)

# plot the image
plt.imshow(im)
                
# some points
x = [100, 100, 400, 400]
y = [200, 500, 400, 500]

# plot the points with red star-markers
plt.plot(x, y, 'r*')
##plot(x,y) # default blue solid line
##plot(x,y,'r*') # red star-markers
##plot(x,y,'go-') # green line with circle-markers
##plot(x,y,'ks:') # black dotted line with square-markers

# line plot
##>>> x[:2]
##[100, 100]
##>>> x[:3]
##[100, 100, 400]
##>>> x[:4]
##[100, 100, 400, 400]
plt.plot(x[:4], y[:4])
plt.plot([150, 200],
         [300,400])


# add title and show the plot
plt.title('Plotting: "empire.jpg"')
plt.show()
