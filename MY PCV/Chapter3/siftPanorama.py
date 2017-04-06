from MYLIBPCV.tools import sift
from MYLIBPCV.tools import homography
from MYLIBPCV.tools import warp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as IM
from scipy.ndimage import affine_transform
import os
from scipy.ndimage import filters
from numpy import *

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()


featname = ['Univ'+str(i)+'.sift' for i in range(1, 6)]
imname = ['Univ'+str(i)+'.jpg' for i in range(1, 6)]

l = {}
d = {}
for i in range(5):
    sift.processImage(os.path.join(dataFolder, imname[i]),\
                      os.path.join(dataFolder, featname[i]))
    l[i],d[i] = sift.readFeaturesFromFile(os.path.join(dataFolder, featname[i]))

matches = {}
for i in range(4):
    matches[i] = sift.match(d[i+1], d[i])

# function to convert the matches to hom. points
def convert_points(j):
    ndx = matches[j].nonzero()[0]
    fp = homography.makeHomog(l[j+1][ndx,:2].T)
    ndx2 = [int(matches[j][i]) for i in ndx]
    tp = homography.makeHomog(l[j][ndx2,:2].T)
    return fp,tp

# estimate the homographies
model = homography.RansacModel()
fp,tp = convert_points(1)

H_12 = homography.H_from_ransac(fp,tp,model)[0] #im 1 to 2
fp,tp = convert_points(0)

H_01 = homography.H_from_ransac(fp,tp,model)[0] #im 0 to 1
tp,fp = convert_points(2) #NB: reverse order

H_32 = homography.H_from_ransac(fp,tp,model)[0] #im 3 to 2
tp,fp = convert_points(3) #NB: reverse order

H_43 = homography.H_from_ransac(fp,tp,model)[0] #im 4 to 3

# warp the images
delta = 2000 # for padding and translation
im1 = np.array(IM.open(os.path.join(dataFolder,imname[1])))
im2 = np.array(IM.open(os.path.join(dataFolder,imname[2])))
im_12 = warp.panorama(H_12,im1,im2,delta,delta)

im1 = np.array(IM.open(os.path.join(dataFolder,imname[0])))
im_02 = warp.panorama(dot(H_12,H_01),im1,im_12,delta,delta)

im1 = np.array(IM.open(os.path.join(dataFolder,imname[3])))
im_32 = warp.panorama(H_32,im1,im_02,delta,delta)

im1 = np.array(IM.open(os.path.join(dataFolder,imname[4])))
im_42 = warp.panorama(dot(H_32,H_43),im1,im_32,delta,2*delta)

plt.gray()
plt.imshow(im_42)
plt.show()
