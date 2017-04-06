import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

def planeSweepNCC(iml, imr, start, steps, wid):
    """ Find disparity image using normalized cross-correlation."""
    m, n = iml.shape

    # arrays to hold the different sums
    meanl = np.zeros((m,n))
    meanr = np.zeros((m,n))
    s = np.zeros((m,n))
    sl = np.zeros((m,n))
    sr = np.zeros((m,n))

    # array to hold depth planes
    dmaps = np.zeros((m,n, steps))

    # compute mean of patch
    filters.uniform_filter(iml, wid, meanl)
    filters.uniform_filter(imr, wid, meanr)

    # normalized images
    norml = iml - meanl
    normr = imr - meanr

    # try different disparities
    for displ in range(steps):
        # move left image to the right, compute sums
        filters.uniform_filter(np.roll(norml,-displ-start)*normr,wid,s) # sum nominator
        filters.uniform_filter(np.roll(norml,-displ-start)*\
                               np.roll(norml,-displ-start),wid,sl)
        filters.uniform_filter(normr*normr,wid,sr) # sum denominator
        # store ncc scores
        dmaps[:,:,displ] = s/np.sqrt(sl*sr)
        
    # pick best depth for each pixel
    return np.argmax(dmaps,axis=2)

def planeSweepGauss(iml, imr, start, steps, wid):
    """ Find disparity image using normalized cross-correlation
    with Gaussian weighted neighborhoods."""
    
    m, n = iml.shape

    # arrays to hold the different sums
    meanl = np.zeros((m,n))
    meanr = np.zeros((m,n))
    s = np.zeros((m,n))
    sl = np.zeros((m,n))
    sr = np.zeros((m,n))

    # array to hold depth planes
    dmaps = np.zeros((m,n, steps))

    # compute mean of patch
    filters.gaussian_filter(iml, wid, 0, meanl)
    filters.gaussian_filter(imr, wid, 0, meanr)

    # normalized images
    norml = iml - meanl
    normr = imr - meanr

    # try different disparities
    for displ in range(steps):
        # move left image to the right, compute sums
        filters.gaussian_filter(np.roll(norml,-displ-start)*normr,wid,0,s) # sum nominator
        filters.gaussian_filter(np.roll(norml,-displ-start)*\
                               np.roll(norml,-displ-start),wid,0,sl)
        filters.gaussian_filter(normr*normr,wid,0,sr) # sum denominator
        # store ncc scores
        dmaps[:,:,displ] = s/np.sqrt(sl*sr)
        
    # pick best depth for each pixel
    return np.argmax(dmaps,axis=2)
    
