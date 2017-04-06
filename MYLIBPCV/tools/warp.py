import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as IM
from MYLIBPCV.tools import homography
from scipy.ndimage import affine_transform
import matplotlib.delaunay as md
from scipy import ndimage
from numpy import *

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

def panorama(H,fromim,toim,padding=2400,delta=2400):
    """ Create horizontal panorama by blending two images
    using a homography H (preferably estimated using RANSAC).
    The result is an image with the same height as toim. 'padding'
    specifies number of fill pixels and 'delta' additional translation.

    For a general geometric_transform() , a function describing the pixel to pixel map needs
    to be specified.

    """
    
    # check if images are grayscale or color
    is_color = len(fromim.shape) == 3
    # homography transformation for geometric_transform()
##    In this case, transf() does this by multiplying with H and normalizing
##    the homogeneous coordinates.
    def transf(p):
        p2 = dot(H,[p[0],p[1],1])
        return (p2[0]/p2[2],p2[1]/p2[2])
    
    if H[1,2]<0: # fromim is to the right
        print ('warp - right')
        
        # transform fromim
        if is_color:
            # pad the destination image with zeros to the right
            toim_t = hstack((toim,zeros((toim.shape[0],padding,3))))
            fromim_t = zeros((toim.shape[0],toim.shape[1]+padding,toim.shape[2]))
            
            for col in range(3):
                fromim_t[:,:,col] = ndimage.geometric_transform(fromim[:,:,col],\
                                                                transf,(toim.shape[0],toim.shape[1]+padding))
        else:
            # pad the destination image with zeros to the right
            toim_t = hstack((toim,zeros((toim.shape[0],padding))))
            fromim_t = ndimage.geometric_transform(fromim,transf,\
                                                   (toim.shape[0],toim.shape[1]+padding))
    else:
        print ('warp - left')
        
        # add translation to compensate for padding to the left
        H_delta = array([[1,0,0],[0,1,-delta],[0,0,1]])
        H = dot(H,H_delta)

        # transform fromim
        if is_color:
            # pad the destination image with zeros to the left
            toim_t = hstack((zeros((toim.shape[0],padding,3)),toim))
            fromim_t = zeros((toim.shape[0],toim.shape[1]+padding,toim.shape[2]))
            for col in range(3):
                fromim_t[:,:,col] = ndimage.geometric_transform(fromim[:,:,col],\
                                                                transf,(toim.shape[0],toim.shape[1]+padding))
        else:
            # pad the destination image with zeros to the left
            toim_t = hstack((zeros((toim.shape[0],padding)),toim))
            fromim_t = ndimage.geometric_transform(fromim,\
                                                   transf,(toim.shape[0],toim.shape[1]+padding))

    # blend and return (put fromim above toim)
    if is_color:
        # all non black pixels
        alpha = ((fromim_t[:,:,0] * fromim_t[:,:,1] * fromim_t[:,:,2] ) > 0)
        for col in range(3):
            toim_t[:,:,col] = fromim_t[:,:,col]*alpha + toim_t[:,:,col]*(1-alpha)
    else:
        alpha = (fromim_t > 0)
        toim_t = fromim_t*alpha + toim_t*(1-alpha)
        
    return toim_t

