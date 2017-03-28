import numpy as np
from numpy import linalg as LA

##>>> np.array([[2,1],[-1,1]])
##array([[ 2,  1],
##       [-1,  1]])
##
##>>> np.array([[-1],[0]])
##array([[-1],
##       [ 0]])
##>>> np.vstack((np.array([[-1],[0]]),np.array([[-1],[0]])))
##array([[-1],
##       [ 0],
##       [-1],
##       [ 0]])
##>>> np.hstack((np.array([[-1],[0]]),np.array([[-1],[0]])))
##array([[-1, -1],
##       [ 0,  0]])

# use eigenvector as basic
##A = LA.matrix_power(np.array([[0, 1], [1, 1]]),2)
##A = np.array([[3, 1], [0, 2]])
A = np.array([[0, 1], [1, 1]])
w, v = LA.eig(A)

v = np.array([[2, 2], [1 + np.sqrt(5), 1 - np.sqrt(5)]])

EIGEN_VECTOR = v
INV_EIGEN_VECTOR = LA.inv(v)

# *** calculation: left to right
TEMP = np.dot(A, EIGEN_VECTOR)
A_IN_EIGEN_BASIS = np.dot(INV_EIGEN_VECTOR, TEMP)

# use eigenvector as basic
CHANGE_BASIS_MATRIX = np.array([[1, -1], [0, 1]])
INV_CHANGE_BASIS_MATRIX = LA.inv(CHANGE_BASIS_MATRIX)
TRANSFORM_MATRIX = np.array([[3, 1], [0, 2]])

##TRANSFORM_MATRIX * CHANGE_BASIC_MATRIX #Change some vector into normal coordinate

TEMP = np.dot(TRANSFORM_MATRIX, CHANGE_BASIS_MATRIX)
TRANSFORM_MATRIX_IN_NEW_BASIS = np.dot(INV_CHANGE_BASIS_MATRIX, TEMP) # Change back to basic matrix

TEST_VECTOR = np.array([[1], [2]])
NORMAL_BASIS = np.dot(LA.matrix_power(TRANSFORM_MATRIX,2), TEST_VECTOR)

TEMP = np.dot(INV_CHANGE_BASIS_MATRIX, TEST_VECTOR)
TEMP_2 = np.dot(LA.matrix_power(TRANSFORM_MATRIX_IN_NEW_BASIS,2), TEMP)
NEW_BASIS = np.dot(CHANGE_BASIS_MATRIX, TEMP_2)




