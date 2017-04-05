from MYLIBPCV.tools import homography
from MYLIBPCV.tools import warp
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# open image to warp
fromim = np.array(Image.open(os.path.join(dataFolder,'sunset_tree.jpg')))
x,y = np.meshgrid(range(5),range(6))
x = (fromim.shape[1]/4) * x.flatten()
y = (fromim.shape[0]/5) * y.flatten()
# triangulate
tri = warp.triangulate_points(x,y)
# open image and destination points
im = np.array(Image.open(os.path.join(dataFolder,'turningtorso1.jpg')))
tp = np.loadtxt(os.path.join(dataFolder,'turningtorso1_points.txt')) # destination points
# convert points to hom. coordinates
fp = np.vstack((y,x,np.ones((1,len(x)))))
tp = np.vstack((tp[:,1],tp[:,0],np.ones((1,len(tp)))))
# warp triangles
im = warp.pwAffine(fromim,im,fp,tp,tri)
# plot
plt.figure()
plt.imshow(im)
warp.plotMesh(tp[1],tp[0],tri)
plt.axis('off')
plt.show()
