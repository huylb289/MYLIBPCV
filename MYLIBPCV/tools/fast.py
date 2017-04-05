import numpy as np
from PIL import Image as IM
import matplotlib.pyplot as plt
import os
import numpy.linalg as LA

fastPath = '~/fast/fast-Linux-x86_64'

""" Ouput file
48 3
51 3
58 3

Each row contains the x-coordinate, y-coordinate.
"""

def processImage(imagename, resultname, \
                 params='-l -t 20 -n 9'):

    cmmd = str(fastPath + ' ' + params + ' ' + imagename\
               + ' ' + resultname)

    os.system(cmmd)
    print ('processed {} to {}'.format(imagename, resultname))

def readFastPointsFromFile(filename):
    """ Rad features properties and return in matrix form"""
    f = np.loadtxt(filename)
    return np.vstack((f[:,1],f[:,0])).T # reverse coordinate x,y coordinates

def plotFastPoints(image, filteredCoords):
    """ Plot corner found in image"""
    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filteredCoords],[p[0] for p in filteredCoords], '*')
    plt.axis('off')
    plt.show()

def getDescriptors(image, filteredCoords, width=5):
    """ For each point return, pixel values around the point using a neighbourhood
    of width 2*width+1. (Assume points are extracted with minDistance > width)"""

    desc = []
    for i in range(0,len(filteredCoords)):
        patch = image[int(filteredCoords[i][0])-width:int(filteredCoords[i][0])+width,\
                      int(filteredCoords[i][1])-width:int(filteredCoords[i][1])+width]

        desc.append(patch)

    return desc

def matchMaximumDistance(desc1, desc2, threshold=0.5):
    """ For each corner point descriptor in the first image,
    select its match to second image using normalized cross-correlation."""

    n = len(desc1[0])

    print("n: {}\n".format(n))
    print("desc1: {}\n".format(len(desc1)))
    print("desc2: {}\n".format(len(desc2)))
    
    # pair-wise distances
    d = -np.ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = desc1[i]**2
            d2 = desc2[j]**2
            compareValue = np.sum(np.sqrt(d1 + d2))
            if compareValue > threshold:
                d[i,j] = compareValue

    # sort desc 
    ndx = np.argsort(-d)
    matchscores = ndx[:,0] # get the first column

    return matchscores

def matchTwoSidedMaximumDistance(desc1, desc2, threshold=0.5):
    """ Two-sided symetric version of match()."""
    matches12 = matchMaximumDistance(desc1, desc2, threshold)
    matches21 = matchMaximumDistance(desc2, desc1, threshold)

##>>> np.where(A > 0.15)
##(array([0, 0, 0, 1, 2, 2, 2]), array([0, 1, 2, 2, 0, 1, 2]))
##>>> np.where(A > 0.15)[0]
##array([0, 0, 0, 1, 2, 2, 2])
    ndx12 = np.where(matches12 > 0) [0]

    print("ndx: {}\n".format(len(ndx12)))

    # remove matches that are not symmetric
    for n in ndx12:
        if matches21[matches12[n]] != n:
            matches12[n] = -1

    return matches12

def match(desc1, desc2, threshold=0.5):
    """ For each corner point descriptor in the first image,
    select its match to second image using normalized cross-correlation."""

    n = len(desc1[0])

    print("n: {}\n".format(n))
    print("desc1: {}\n".format(len(desc1)))
    print("desc2: {}\n".format(len(desc2)))
    
    # pair-wise distances
    d = -np.ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            try:
                d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
                d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
                nccValue = np.sum(d1 * d2) / (n - 1)
                if nccValue > threshold:
                    d[i,j] = nccValue
            except:
                continue

    # sort desc 
    ndx = np.argsort(-d)
    matchscores = ndx[:,0] # get the first column

    return matchscores

def matchTwoSided(desc1, desc2, threshold=0.5):
    """ Two-sided symetric version of match()."""
    matches12 = match(desc1, desc2, threshold)
    matches21 = match(desc2, desc1, threshold)

##>>> np.where(A > 0.15)
##(array([0, 0, 0, 1, 2, 2, 2]), array([0, 1, 2, 2, 0, 1, 2]))
##>>> np.where(A > 0.15)[0]
##array([0, 0, 0, 1, 2, 2, 2])
    ndx12 = np.where(matches12 > 0) [0]

    print("ndx: {}\n".format(len(ndx12)))

    # remove matches that are not symmetric
    for n in ndx12:
        if matches21[matches12[n]] != n:
            matches12[n] = -1

    return matches12

def appendimages(im1,im2):
    """ Return a new image that appends the two images side-by-side. """
    
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    
    if rows1 < rows2:
        im1 = np.concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
        
    # if none of these cases they are equal, no filling needed.
    return np.concatenate((im1,im2), axis=1)

def plotMatches(im1,im2,locs1,locs2,matchscores,show_below=True):
    """ Show a figure with lines joining the accepted matches
    input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
    matchscores (as output from 'match()'),
    show_below (if images should be shown below matches). """
    im3 = appendimages(im1,im2)
    if show_below:
        im3 = np.vstack((im3,im3))

    plt.imshow(im3)
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plt.plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
    plt.axis('off')
        
