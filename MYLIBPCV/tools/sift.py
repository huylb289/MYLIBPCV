import numpy as np
from PIL import Image as IM
import matplotlib.pyplot as plt
import os
import numpy.linalg as LA

siftPath = '~/vlfeat/bin/glnxa64/sift'

""" Ouput file
318.861 7.48227 1.12001 1.68523 0 0 0 1 0 0 0 0 0 11 16 0 ...

Each row contains the coordinates, scale, and rotation angle for each interest point
as the first four values. followed by the 128 values of the corresponding descriptor.

"""

def processImage(imagename, resultname, \
                 params="--edge-thresh 10 --peak-thresh 5"):
    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = IM.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    cmmd = str(siftPath + ' ' + imagename + ' --output='+ resultname +
               ' ' + params)

    os.system(cmmd)
    print ('processed {} to {}'.format(imagename, resultname))

def readFeaturesFromFile(filename):
    """ Rad features properties and return in matrix form"""
    f = np.loadtxt(filename)
    return f[:,:4], f[:,4:] # feature locations, descriptor

def writeFeaturesToFile(filename, locs, desc):
    """ Save feature location and descriptor to file. """
    np.savetxt(filename, np.hstack((locs, desc)))

def plotFeatures(im, locs, circle=False, direction=False):
    """ Show image with features.
    input: im (image as array)
    locs: (row, col, scale, orientation of each feature)."""

    # By parametric equation
    # http://jwilson.coe.uga.edu/EMAT6680Fa05/Parveen/Assignment%2010/parametric_equations.htm
    def drawCircle(c, r):
        t = np.arange(0,1.01, 0.01)*2*np.pi
        x = r*np.cos(t) + c[0]
        y = r*np.sin(t) + c[1]
        plt.plot(x, y, 'b', linewidth=1)

    plt.imshow(im)
    if circle:
        for p in locs:
            drawCircle(p[:2],p[2])
            if direction:
                print("### Plotting direction")
    else:
        plt.plot(locs[:,0],locs[:,1],'ob')
    
    plt.axis('off')

"""
A robust criteria (also introduced by Lowe) for matching a feature in one image to a
feature in another image is to use the ratio of the distance to the two closest match-
ing features
"""
def match(desc1, desc2):
    """ For each descriptor in the first image,
    select its match in the second image.
    input: desc1 (descriptors for the first image),
    desc2 (same for second image). """

    # normalization d/norm (norm: sqrt(a**2 + b**2 ...) )
    desc1 = np.array([d/LA.norm(d) for d in desc1])
    desc2 = np.array([d/LA.norm(d) for d in desc2])

    distRatio = 0.6
    desc1Size = desc1.shape

    """
    >>> A = np.random.rand(3,1)
    >>> A
    array([[ 0.63122864],
           [ 0.6964063 ],
           [ 0.57636382]])
    >>> for i,m in enumerate(A):
            print(i,m)

            
    0 [ 0.63122864]
    1 [ 0.6964063]
    2 [ 0.57636382]
    >>> A = np.random.rand(3,)
    >>> A
    array([ 0.28300844,  0.12998464,  0.31607506])
    >>> for i,m in enumerate(A):
            print(i,m)

            
    0 0.283008443027
    1 0.129984642352
    2 0.316075060561
    """
##    matchscores = np.zeros((desc1Size[0], 1), 'int') # ERROR
    matchscores = np.zeros((desc1Size[0]), 'int')
    desc2t = desc2.T # precompute matrix transpose
    for i in range(desc1Size[0]):
        """
        >>> A
        array([[-0.0910809 ,  0.78402878,  0.3626241 ],
               [ 0.30073073,  0.39571111,  1.49319532],
               [ 0.56398206, -0.972467  ,  1.14193303]])
        >>> A[1,:]
        array([ 0.30073073,  0.39571111,  1.49319532])
        """
        dotprods = np.dot(desc1[i,:], desc2t) # vector of dot products
        dotprods = 0.9999*dotprods

        # inverse cosine and sort, return index for features in second image
        indx = np.argsort(np.arccos(dotprods))

        # check if nearest neighbor has angle less than distRatio times 2nd
        if np.arccos(dotprods)[indx[0]] < \
           distRatio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = np.int(indx[0])

    return matchscores

def matchTwoSided(desc1, desc2):
    """ Two-sided symmetric version of match()"""

    matches12 = match(desc1, desc2)
    matches21 = match(desc2, desc1)

    ndx12 = matches12.nonzero()[0]

    # remove matches that are not symmetric
    for n in ndx12:
        if matches21[int(matches12[n])] != n:
            matches12[n] = 0

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
            plt.plot([locs1[i][0],locs2[m][0]+cols1],[locs1[i][1],locs2[m][1]],'c')
    plt.axis('off')
