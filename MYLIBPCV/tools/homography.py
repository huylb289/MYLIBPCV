import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as IM
from numpy import *

def normalize(points):
    """ Normalize a collection of points in homogeneous
    coordinates so that the last row is 1"""

    for row in points:
        row /= points[-1]

    return points

def makeHomog(points):
    """ Convert a  set of points (dim*n array) to
    homogeneous coordinates."""

    return np.vstack((points, np.ones((1, points.shape[1]))))

def HFromPoints(fp, tp):
    """ Find Homography H, such that fp is mapped to tp
    using the linear DLT method. Points are conditioned automatically."""

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # condition points (important for numerical reasons)
    # --from points--
    m = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = np.dot(C1, fp)

    # --to points--
    m = np.mean(tp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
    C2 = np.diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = np.dot(C2, tp)

    # Direct Linear Transformation - DLT 
    # create matrix for linear method, 2 rows for each correspondence pair
    nbrCorrespondences = fp.shape[1]
    A = np.zeros((2*nbrCorrespondences,9))
    for i in range(nbrCorrespondences):
        A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,\
                  tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
        A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,\
                    tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]

    U,S,V = np.linalg.svd(A)
    H = V[8].reshape((3,3))

    # decondition
    H = dot(linalg.inv(C2),dot(H,C1))
    
    # normalize and return
    return H / H[2,2]

def HaffineFromPoints(fp, tp):
    """ Find H, affine transformation, such that
    tp is affine transf of fp."""

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # condition points
    # --from points--
    m = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fpCond = np.dot(C1, fp)

    # --to points--
    m = np.mean(tp[:2], axis=1)
    maxstd = np.max(np.std(tp[:2], axis=1)) + 1e-9
    C2 = np.diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tpCond = np.dot(C2, tp)

    # conditioned points have mean zero, so translation is zero
    A = np.concatenate((fpCond[:2],tpCond[:2]), axis=0)
    U,S,V = np.linalg.svd(A.T)

    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros((2,1))), axis=1)
    H = np.vstack((tmp2, [0,0,1]))

    # decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    return H / H[2,2]

class RansacModel(object):
    """ Class for testing homography fit with ransac.py from
    http://www.scipy.org/Cookbook/RANSAC"""
    
    def __init__(self,debug=False):
        self.debug = debug
        
    def fit(self, data):
        """ Fit homography to four selected correspondences. """
        
        # transpose to fit H_from_points()
        data = data.T
        
        # from points
        fp = data[:3,:4]
        
        # target points
        tp = data[3:,:4]
        
        # fit homography and return
        return HFromPoints(fp,tp)

    def get_error(self, data, H):
        """ Apply homography to all correspondences,
        return error for each transformed point. """
        
        data = data.T
        
        # from points
        fp = data[:3]
        
        # target points
        tp = data[3:]
        
        # transform fp
        fpTransformed = np.dot(H,fp)
        
        # normalize hom. coordinates
        for i in range(3):
            fpTransformed[i] /= fpTransformed[2]
            
        # return error per point
        return np.sqrt( np.sum((tp-fpTransformed)**2,axis=0) )

def H_from_ransac(fp,tp,model,maxiter=1000,match_theshold=10):
    """ Robust estimation of homography H from point
    correspondences using RANSAC (ransac.py from
    http://www.scipy.org/Cookbook/RANSAC).
    input: fp,tp (3*n arrays) points in hom. coordinates. """
    
    from MYLIBPCV.tools import ransac
    
    # group corresponding points
    data = vstack((fp,tp))
    
    # compute H and return
    H,ransacData = ransac.ransac(data.T,model,4,maxiter,match_theshold,10,\
                                  return_all=True)
    
    return H,ransacData['inliers']
