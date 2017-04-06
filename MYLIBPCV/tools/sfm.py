import numpy as np
import matplotlib.pyplot as plt

"""
As usual, we compute the least squares solution using SVD. Since the resulting solution
might not have rank 2 as a proper fundamental matrix should, we replace the result
with the closest rank 2 approximation by zeroing out the last singular value.
"""
def computeFundamental(x1, x2):
    """ Computes the fundamental matrix from corresponding points
    (x1,x2 3*n arrays) using the normalized 8 point algorithm.
    each row is constructed as
    [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # build matrix for equations
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i]]

    # compute linear least square solution
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S),V))

    return F

def computeEpipole(F):
    """ Computes the (right) epipole from a
    fundamental matrix F.
    (Use with F.T for left epipole."""

    # return null space of F (Fx = 0)
    U,S,V = np.linalg.svd(F)
    e = V[-1]
    return e/e[2]

def plotEpipolarLine(im, F, x, epipole=None, showEpipole=True):
    """ Plot the epipole and epipolar line F*x = 0
    in an image. F is the fundamental matrix and x a point in the other image"""

    m,n = im.shape[:2]
    line = np.dot(F,x)

    # epipolar line parameter and values
    t = np.linspace(0, n, 100)
    lt = np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

    # take only line points inside the image
    ndx = (lt >= 0) & (lt < m)
    plt.plot(t[ndx], lt[ndx], linewidth=2)

    if showEpipole:
        if epipole is None:
            epipole = computeEpipole(F)
        plt.plot(epipole[0]/epipole[2], epipole[1]/epipole[2], 'r*')

def triangulatePoint(x1, x2, P1, P2):
    """ Point pair triangulation from
    least squares solution."""

    M = np.zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2
    
    U,S,V = np.linalg.svd(M)
    X = V[-1,:4]
    return X / X[3]

def triangulate(x1,x2,P1,P2):
    """ Two-view triangulation of points in
        x1,x2(3*n homog. coordinates)."""

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match")

    X = [triangulatePoint(x1[:,i], x2[:,i], P1, P2) for i in range(n)]
    return np.array(X)

def computeP(x,X):
    """ Compute camera matrix from pairs of
    2D-3D correspondences (in homog. coordinates)."""

    n = x.shape[1]
    if X.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # create matrix for DLT solution
    M = np.zeros((3*n, 12+n))
    for i in range(n):
        M[3*i, 0:4] = X[:,i]
        M[3*i+1, 4:8] = X[:,i]
        M[3*i+2, 8:12] = X[:,i]
        M[3*i:3*i+3, i+12] = -x[:,i]

    U,S,V = np.linalg.svd(M)
    return V[-1,:12].reshape((3,4))

def skew(a):
    """ Skew matrix A such that a x v = Av for any v."""
    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
        
def computePFromFundamental(F):
    """ Computes the second camera matrix (assuming P1=[I 0])
        from a fundamental matrix."""

    e = computeEpipole(F.T) # left epipole
    Te = skew(e)
    return np.vstack((np.dot(Te, F.T).T, e)).T

"""
The fundamental matrix for calibration-normalized coordinates is called the essential
matrix and is usually denoted E instead of F
"""
def computePFromEssential(E):
    """ Computes the second camera matrix (assuming P1 = [I 0])
    from an essential matrix. Output is a list of four
    possible camera matrices. """

    # make sure E is rank 2
    U,S,V = np.linalg.svd(E)
    if np.linalg.det(dot(U,V))<0:
        V = -V
        
    E = np.dot(U,dot(diag([1,1,0]),V))
    # create matrices (Hartley p 258)
    Z = skew([0,0,-1])
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    # return all four solutions
    P2 = [np.vstack((dot(U,dot(W,V)).T,U[:,2])).T,\
          np.vstack((dot(U,dot(W,V)).T,-U[:,2])).T,\
          np.vstack((dot(U,dot(W.T,V)).T,U[:,2])).T,\
          np.vstack((dot(U,dot(W.T,V)).T,-U[:,2])).T]
    
    return P2

"""
Multiple View Reconstruction
"""

def compute_fundamental_normalized(x1,x2):
    """ Computes the fundamental matrix from corresponding points
    (x1,x2 3*n arrays) using the normalized 8 point algorithm. """
    
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    
    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = np.dot(T1,x1)
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = np.dot(T2,x2)
    
    # compute F with the normalized coordinates
    F = computeFundamental(x1,x2)
    
    # reverse normalization
    F = np.dot(T1.T,np.dot(F,T2))
    return F/F[2,2]


class RansacModel(object):
    """ Class for fundmental matrix fit with ransac.py from
    http://www.scipy.org/Cookbook/RANSAC"""
    
    def __init__(self,debug=False):
        self.debug = debug

    def fit(self, data):
        """ Estimate fundatmental matrix using eight
        selected correspondences"""

        # transpose and split data into the two points sets
        data = data.T
        x1 = data[:3, :8]
        x2 = data[3:, :8]

        # estimate fundamental matrix and return
        F = compute_fundamental_normalized(x1,x2)

        return F
    
    def get_error(self, data, F):
        """ Compute x^T F x for all correspondences
        return error for each transformed point."""

        # transpose and split data into the two point
        data = data.T
        x1 = data[:3]
        x2 = data[3:]


        # Sampson distance as error measure
        Fx1 = np.dot(F,x1)
        Fx2 = np.dot(F,x2)
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        err = (np.diag(np.dot(x1.T, np.dot(F, x2)))) **2 / denom

        # return error per point
        return err

def F_from_ransac(x1,x2,model, maxiter=5000, match_threshold=1e-6):
    """ Robust estimation of a fundamental matrix F from point
    correspondences using RANSAC (ransac.py from
    http://www.scipy.org/Cookbook/RANSAC).
    
    input: x1,x2 (3*n arrays) points in hom. coordinates. """

    from MYLIBPCV.tools import ransac

    data = np.vstack((x1,x2))

    # compute F and return with inlier index
    F, ransac_data = ransac.ransac(data.T, model, 8, maxiter, match_theshold,\
                                   20, return_all=True)

    return F, ransac_data['inliers']


