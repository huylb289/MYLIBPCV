from MYLIBPCV.tools import homography
from MYLIBPCV.tools import sfm
from MYLIBPCV.tools import sift
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image as IM

dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# calibration
K = np.array([[2394,0,932],[0,2398,628],[0,0,1]])

# load images and compute features
im1Path = os.path.join(dataFolder, 'alcatraz1.jpg')
sift.processImage(im1Path,'im1.sift')
im1 = np.array(IM.open(im1Path))
l1, d1 = sift.readFeaturesFromFile('im1.sift')

im2Path = os.path.join(dataFolder, 'alcatraz2.jpg')
sift.processImage(im2Path,'im2.sift')
im2 = np.array(IM.open(im2Path))
l2, d2 = sift.readFeaturesFromFile('im2.sift')

# match features
matches = sift.matchTwoSided(d1,d2)
ndx = matches.nonzero()[0]

# make homogeneous and normalize with inv(K)
x1 = homography.makeHomog(l1[ndx,:2].T)
ndx2 = [int(matches[i]) for i in ndx]
x2 = homography.makeHomog(l2[ndx2,:2].T)

x1n = np.dot(np.linalg.inv(K), x1)
x2n = np.dot(np.linalg.inv(K), x2)

# estimate E with RANSAC
model = sfm.RansacModel()
E,inliers = sfm.F_from_ransac(x1n,x2n,model)

# compute camera matrices (P2 will be list of four solutions)
P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
P2 = sfm.computePFromEssential(E)


# pick the solution with points in front of cameras
ind = 0
maxres = 0
for i in range(4):
    # triangulate inliers and compute depth for each camera
    X = sfm.triangulate(x1n[:,inliers],x2n[:,inliers],P1,P2[i])
    d1 = np.dot(P1,X)[2]
    d2 = np.dot(P2[i],X)[2]
    if np.sum(d1>0)+np.sum(d2>0) > maxres:
        maxres = np.sum(d1>0)+ np.sum(d2>0)
        ind = i
        infront = (d1>0) & (d2>0)
# triangulate inliers and remove points not in front of both cameras
X = sfm.triangulate(x1n[:,inliers],x2n[:,inliers],P1,P2[ind])
X = X[:,infront]

# 3D plot
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()

ax = fig.gca(projection='3d')
ax.plot(-X[0],X[1],X[2],'k.')

plt.axis('off')


# plot the projection of X
from MYLIBPCV.tools import camera

# project 3D points
cam1 = camera.Camera(P1)
cam2 = camera.Camera(P2[ind])
x1p = cam1.project(X)
x2p = cam2.project(X)

# reverse K normalization
x1p = np.dot(K,x1p)
x2p = np.dot(K,x2p)

plt.figure()
plt.imshow(im1)
plt.gray()
plt.plot(x1p[0],x1p[1],'o')
plt.plot(x1[0],x1[1],'r.')
plt.axis('off')

plt.figure()
plt.imshow(im2)
plt.gray()
plt.plot(x2p[0],x2p[1],'o')
plt.plot(x2[0],x2[1],'r.')
plt.axis('off')

plt.show()
