from scipy import linalg
import numpy as np

class Camera(object):
    """ Class for representing pin-hole camera"""
    def __init__(self, P):
        """ Initialize P = K[R|t] camera model"""
        self.P = P
        self.K = None # calibration matrix
        self.R = None # rotation
        self.t = None # translation
        self.c = None # camera center

    def project(self, X):
        """ Project points in X(4*n array) and normalize coordinates. """
        x = np.dot(self.P, X)
        for i in range(3):
            x[i] /= x[2]

        return x

    def rotation_matrix(a):
        """ Creates a 3D rotation matrix for rotation around the axis of the vector a"""
        """
        Euler’s identity (exp(i*theta) = cos(theta) + i*sin(theta)) applied to a matrix
        >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
        >>> expm(1j*a)
        array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
               [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
        >>> cosm(a) + 1j*sinm(a)
        array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
               [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
        """

        # Rodrigue's Formula
        R = np.eye(4)
        R[:3, :3] = linalg.expm([[0, -a[2], a[1]],
                                 [a[2],0, -a[0]],
                                 [-a[1], a[0], 0]])

        return R

    def factor(self):
        """ Factorize the camera matrix into K, R, t as P = K[P|t].

        In this case, we will use a type of matrix
        factorization called RQ-factorization.                
        """
        # fator first 3*3 part
        K, R = linalg.rq(self.P[:,:3])

        # make diagonal
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1,1] *= -1

        self.K = np.dot(K,T)
        self.R = np.dot(T,R)
        self.t = np.dot(linalg.inv(self.K), self.P[:,3])

        return self.K, self.R, self.t

    def center(self):
        """ Computer Camera center
        Given camera projection matrix, P, and the camera center, C.
        PC = 0. For P = K[R|t]. This gives
        K[R|t]C = K*R*C + K*t = 0
        Camera center can be computed as
        C = -R.T * t"""

        if self.c is not None:
            return self.c
        else:
            # Computer c by factoring
            self.factor()
            self.c = -np.dot(self.R.T,t)
            return self.c

    def my_calibration(sz):
        """ sz: image size
        This function then takes a size tuple and returns the calibration matrix. Here we assume
        the optical center to be the center of the image. Go ahead and replace the focal lengths
        with their mean if you like; for most consumer type cameras this is fine. Note that the
        calibration is for images in landscape orientation. For portrait orientation, you need
        to interchange the constants.


        For the particular setup in Figure 4-3, the object was measured to be 130 by 185 mm,
        so dX = 130 and dY = 185. The distance from camera to object was 460 mm, so
        dZ = 460. You can use any unit of measurement; only the ratios of the measurements
        matter. Using ginput() to select four points in the image, the width and height in pixels
        was 722 and 1040. This means that dx = 722 and dy = 1040. Putting these values in
        the relationship above gives
        fx = 2555, fy = 2586

        Now, it is important to note that this is for a particular image resolution. In this case, the
        image was 2592 × 1936 pixels(width x height)
        """
        row, col = sz
        fx = 2555*col/2592
        fy = 2586*row/1936
        K = np.diag([fx,fy,1])
        K[0,2] = 0.5*col
        K[1,2] = 0.5*row
        return K
