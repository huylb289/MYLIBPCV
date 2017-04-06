from MYLIBPCV.tools import camera
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as IM

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# load some images
im1Path = os.path.join(dataFolder, 'multi-view datasets', \
                       'Merton', 'images', '001.jpg')

im1 = np.array(IM.open(im1Path))

im2Path = os.path.join(dataFolder, 'multi-view datasets', \
                       'Merton', 'images', '002.jpg')

im2 = np.array(IM.open(im2Path))

# load 2D points for each view to a list
points2DPath = os.path.join(dataFolder, 'multi-view datasets', \
                       'Merton', '2D')
points2D= [np.loadtxt(os.path.join(points2DPath, '00'+ str(i) + '.corners')).T \
           for i in range(1,4)]

# load 3D points
points3DPath = os.path.join(dataFolder, 'multi-view datasets', \
                       'Merton', '3D', 'p3d')
points3D = np.loadtxt(points3DPath).T

# load correspondences
corrPath = os.path.join(dataFolder, 'multi-view datasets', \
                        'Merton', '2D', 'nview-corners')
                        
corr = np.genfromtxt(corrPath, dtype='int', missing_values='*')

# load cameras to a list of Camera object
PPath = os.path.join(dataFolder, 'multi-view datasets', \
                       'Merton', '2D')
P = [camera.Camera(np.loadtxt(os.path.join(PPath, '00' + str(i) + '.P')))\
                   for i in range(1,4)]

# make 3D points homogenous and project
X = np.vstack((points3D, np.ones(points3D.shape[1])))
x = P[0].project(X)
