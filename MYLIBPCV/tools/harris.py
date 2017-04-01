from PIL import Image as IM
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

def computeHarrisResponse(im, sigma=3):
    """ Compute the Harris corner detector response function
    for each pixel in a graylevel image"""

    # derivatives
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0,1), imx)

    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1,0), imy)

    # compute components of the Harris matrix
    # from fomular (2.1) page 29 in the book
    # "Programming Computer Vision with Python - Jan Erik Solem"
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    # determinant and trace
    Wdeterminant = Wxx*Wyy - (Wxy**2)
    Wtrace = Wxx + Wyy

    return Wdeterminant / Wtrace

def getHarrisPoints(harrisim, minDist=10, threshold=0.1):
    """ Return corners from a Harris response image
    min_dist is the minimum number of pixels separating
    corners and image boundary. """

    # find top corner candidates above a threshold
    cornerThreshold = harrisim.max() * threshold
    harrisim_t = (harrisim > cornerThreshold) * 1

    # get coordinates of candidates
    coords = np.array(harrisim_t.nonzero()).T
    # their values
    candidateValues = [harrisim[c[0],c[1]] for c in coords]

    # sort candidates
##    >>> x = np.array([3, 1, 2])
##    >>> np.argsort(x)
##    array([1, 2, 0])
    index = np.argsort(candidateValues)

    # store allowed point locations in array
##>>> A = np.random.rand(5,5)
##>>> A
##array([[ 0.92051466,  0.5810133 ,  0.29408974,  0.87221451,  0.81752013],
##       [ 0.12215516,  0.48137907,  0.33152909,  0.51876608,  0.37834504],
##       [ 0.71962784,  0.17070431,  0.23710843,  0.27570105,  0.71211124],
##       [ 0.15123192,  0.63528979,  0.03601388,  0.25650442,  0.01220797],
##       [ 0.05309662,  0.59374243,  0.03383327,  0.18068062,  0.24960569]])
##>>> A[1:-1,1:-1]
##array([[ 0.48137907,  0.33152909,  0.51876608],
##       [ 0.17070431,  0.23710843,  0.27570105],
##       [ 0.63528979,  0.03601388,  0.25650442]])
    allowedLocations = np.zeros(harrisim.shape)
    allowedLocations[minDist:-minDist, minDist:-minDist] = 1

    # select the best points taking min distance into account
    filteredCoords = []
    for i in index:
        if allowedLocations[coords[i,0],coords[i,1]] == 1:
            filteredCoords.append(coords[i])
            allowedLocations[(coords[i,0]-minDist):(coords[i,0]+minDist),
                              (coords[i,1]-minDist):(coords[i,1]+minDist)] = 0

    return filteredCoords

def plotHarrisPoints(image, filteredCoords):
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
    for coords in filteredCoords:
        patch = image[coords[0]-width:coords[0]+width,\
                      coords[1]-width:coords[1]+width]

        desc.append(patch)

    return desc

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
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            nccValue = np.sum(d1 * d2) / (n - 1)
            if nccValue > threshold:
                d[i,j] = nccValue

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

    plt.figure()
    plt.gray()
    plt.imshow(im3)
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plt.plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
    plt.axis('off')
    plt.show()
        
