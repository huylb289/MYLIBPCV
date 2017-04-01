import os 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as IM

def getImlist(path):
    """ Returns a list of filenames for
    all jpg images in a directory."""

    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

def imresize(im, sz):
    """ Resize an image array using PIL """
    pilIm = Image.fromarray(np.uint8(im))

    return np.array(pilIm.resize(sz))

def histeq(im, nbrBin=256):
    """ Histogram equalization of a grayscale image.
    Using function histogram from matplotlib
    
   The function takes a grayscale image and the number of bins to use in the histogram
as input, and returns an image with equalized histogram together with the cumulative
distribution function used to do the mapping of pixel values
    """
    # get image histogram
    imhist, bins = np.histogram(im.flatten(), nbrBin, normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf/cdf[-1] # normalize
##    Note the use of the last
##    element (index -1) of the cdf to normalize it between 0 . . . 1

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf
    
def computeAverage(imlist):
    """ Compute the average of a list of images. """

    # open first image and make into array of type float
    averageim = np.array(IM.open(imlist[0]), 'f')

    for imname in imlist[1:]:
        try:
            averageim += np.array(IM.open(imname))
        except:
            print('imname: {} skipped'.format(imname))

    # ERROR: there are some image not add, but divide by the total imlist
    averageim /= len(imlist)

    # return average as uint8
    return np.array(averageim, 'uint8')
                
        
