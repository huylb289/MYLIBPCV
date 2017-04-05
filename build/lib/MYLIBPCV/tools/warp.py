import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as IM
from MYLIBPCV.tools import homography
from scipy.ndimage import affine_transform
import matplotlib.delaunay as md

def alphaForTriangle(points,m,n):
    """ Creates alpha map of size (m,n)
    for a triangle with corners defined by points
    (given in normalized homogeneous coordinates). """
    
    alpha = np.zeros((m,n))
    for i in range(int(min(points[0])),int(max(points[0]))):
        for j in range(int(min(points[1])),int(max(points[1]))):
            x = np.linalg.solve(points,[i,j,1])
            if min(x) > 0: #all coefficients positive
                alpha[i,j] = 1
            
    return alpha


def imageInImage(im1, im2, tp):
    """ Put im1 in im2 with an affine transformation
    such that corners are as close to tp as possible.
    tp are homogeneous and counterclockwise from top left."""

##    # points to warp from
##    m, n = im1.shape[:2]
##    fp = np.array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])
##
##    # compute affine transform and apply
##    H = homography.HaffineFromPoints(tp, fp)
##    im1T = affine_transform(im1, H[:2,:2],\
##                                    (H[0,2],H[1,2]), im2.shape[:2])
##    alpha = (im1T > 0)
##
##    return (1-alpha)*im2 + alpha*im1T

    # set from points to corners of im1
    m,n = im1.shape[:2]
    fp = np.array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])
    
    # first triangle
    tp2 = tp[:,:3]
    fp2 = fp[:,:3]
    
    # compute H
    H = homography.HaffineFromPoints(tp2,fp2)
    im1_t = affine_transform(im1,H[:2,:2],
    (H[0,2],H[1,2]),im2.shape[:2])
    
    # alpha for triangle
    alpha = alphaForTriangle(tp2,im2.shape[0],im2.shape[1])
    im3 = (1-alpha)*im2 + alpha*im1_t
    
    # second triangle
    tp2 = tp[:,[0,2,3]]
    fp2 = fp[:,[0,2,3]]

    # compute H
    H = homography.HaffineFromPoints(tp2,fp2)
    im1_t = affine_transform(im1,H[:2,:2],
    (H[0,2],H[1,2]),im2.shape[:2])
    # alpha for triangle
    alpha = alphaForTriangle(tp2,im2.shape[0],im2.shape[1])
    im4 = (1-alpha)*im3 + alpha*im1_t

    return im4

def triangulate_points(x,y):
    """ Delaunay triangulation of 2D points. """
    centers,edges,tri,neighbors = md.delaunay(x,y)
    return tri

def pwAffine(fromim, toim, fp, tp, tri):
    """ Warp triangular patches from an image.
    fromim = image to warp
    toim = destination image
    fp = from points in hom. coordinates
    tp = to points in hom. coordinates
    tri = triangulation. """
    
    im = toim.copy()

    # check if image is grayscale or color
    isColor = len(fromim.shape) == 3

    # create image to warp to(needed if iterate colors)
    imT = np.zeros(im.shape, 'uint8')

    for t in tri:
        # compute affine transformation
        H = homography.HaffineFromPoints(tp[:,t],fp[:,t])

        if isColor:
            for col in range(fromim.shape[2]):
                imT[:,:,col] = affine_transform(\
                    fromim[:,:,col],H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])
        else:
            imT = transform(\
                fromim,H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])

        # alpha for triangle
        alpha = alphaForTriangle(tp[:,t], im.shape[0], im.shape[1])

        # add triangle to image
        im[alpha > 0] = imT[alpha>0]

    return im

def plotMesh(x,y,tri):
    """ Plot triangles. """

    for t in tri:
        tExt = [t[0], t[1], t[2], t[0]] # add first point to end
        plt.plot(x[tExt],y[tExt],'r')
