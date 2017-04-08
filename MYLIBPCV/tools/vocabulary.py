from scipy.cluster import vq
from MYLIBPCV.tools import sift
import numpy as np

class Vocabulary(object):
    def __init__(self, name):
        self.name = name
        self.voc = []
        self.idf = []
        self.trainingdata = []
        self.nbrWords = 0

    def train(self, featurefiles, k=100, subsampling=10):
        """ Train a vocabulary from features in files listed
        in featurefiles using k-means with k number of words.
        Subsampling of training data can be used for speedup."""

        nbrImages = len(featurefiles)

        # read the features from file
        descr = []
        descr.append(sift.readFeaturesFromFile(featurefiles[0])[1])
        descriptors = descr[0] # stack all features for k-means
        for i in np.arange(1, nbrImages):
            descr.append(sift.readFeaturesFromFile(featurefiles[i])[1])
            descriptors = np.vstack((descriptors, descr[i]))

        # k-means: last number determines number of runs
        self.voc, distortion = vq.kmeans(descriptors[::subsampling, :], k, 1)
        self.nbrWords = self.voc.shape[0]

        # go through all training images and project on vocabulary
        imwords = np.zeros((nbrImages, self.nbrWords))
        for i in range(nbrImages):
            imwords[i] = self.project(descr[i])

        nbrOccurences = np.sum((imwords > 0)*1, axis=0)

        self.idf = np.log((1.0*nbrImages) / (1.0*nbrOccurences))
        self.trainingdata = featurefiles


    def project(self,descriptors):
        """ Project descriptors on the vocabulary
        to create a histogram of words. """
        
        # histogram of image words
        imhist = np.zeros((self.nbrWords))
        words,distance = vq.vq(descriptors,self.voc)
        for w in words:
            imhist[w] += 1
        return imhist

    def candidates_from_histogram(self,imwords):
        """ Get list of images with similar words. """
        
        # get the word ids
        words = imwords.nonzero()[0]
        
        # find candidates
        candidates = []
        for word in words:
            c = self.candidates_from_word(word)
            candidates+=c
            
        # take all unique words and reverse sort on occurrence
        tmp = [(w,candidates.count(w)) for w in set(candidates)]
        tmp.sort(cmp=lambda x,y:cmp(x[1],y[1]))
        tmp.reverse()
        
        # return sorted list, best matches first
        return [w[0] for w in tmp]
        
