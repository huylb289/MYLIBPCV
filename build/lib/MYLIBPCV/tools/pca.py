from PIL import Image as IM
import numpy as np
from numpy import linalg as LA

def pca(X):
    """
    Principle Component Analysis
    
    input: X, matrix with training data stored as flattened array in rows
    output: projection matrix (with important dimensions first), variance and mean.
    """
    # get dimensions
    num_data, dim = X.shape

    # center data
    ##    Assuming zero mean data (subtract the mean), consider the
    ##indexed vectors {x 1, x 2, ..., x m } which are the rows of an mxn
    ##matrix X.
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA - compact trick data
        M = np.dot(X, X.T) # compute covariance matrix, to compute eigen vector
        # because covariance always has form of nxn
        # so it always exist eigenvectors, and these eigenvectors always
        # perpendicular with each other, no matter how much dimensions we have.
        
        ##All eigenvectors of a symmetric(covariance) * matrix are
        ##perpendicular to each other, no matter how many
        ##dimensions we have.
        
        e, EV = LA.eigh(M) # eigenvalues, and eigenvectors
        tmp = X.T.dot(EV) # the compact trick
        V = tmp[::-1] # reverse since the last eigenvectors are the ones we want
        S = np.sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = LA.linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean
    return V,S,mean_X
        
