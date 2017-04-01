import os 
import numpy as np
import matplotlib.pyplot as plt


def get_imlist(path):
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
    imhist, bins = plt.histogram(im.flatten(), nbrBin, normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf/cdf[-1] # normalize
##    Note the use of the last
##    element (index -1) of the cdf to normalize it between 0 . . . 1

    # use linear interpolation of cdf to find new pixel values
    im2 = plt.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf
    
