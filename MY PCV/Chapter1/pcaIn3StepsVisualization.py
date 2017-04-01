import pandas as pd

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()

# split data table into data X and class labels y

X = df.ix[:,0:4].values
y = df.ix[:,4].values


import matplotlib.pyplot as plt
import numpy as np
import math

##label_dict = {1: 'Iris-setosa',
##              2: 'Iris-versicolor',
##              3: 'Iris-virginica'}
##
##feature_dict = {0: 'sepal length [cm]',
##                1: 'sepal width [cm]',
##                2: 'petal length [cm]',
##                3: 'petal width [cm]'}
##
##with plt.style.context('seaborn-whitegrid'):
##    plt.figure(figsize=(8,6))
##    for cnt in range(4):
##        plt.subplot(2,2, cnt+1)
##        for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
##            plt.hist(X[y==lab, cnt],
##                     label=lab,
##                     bins=10,
##                     alpha=0.3)
##
##        plt.xlabel(feature_dict[cnt])
##
##    plt.legend(loc='upper right', fancybox=True, fontsize=8)
##
##    plt.tight_layout()
##    plt.show()

from sklearn.preprocessing import StandardScaler

##Whether to standardize the data prior to a PCA on the covariance matrix
##depends on the measurement scales of the original features.
##Since PCA yields a feature subspace that maximizes the variance along the axes,
##it makes sense to standardize the data, especially,
##if it was measured on different scales.

X_std = StandardScaler().fit_transform(X)

##1 - Eigendecomposition - Computing Eigenvectors and Eigenvalues
##The eigenvectors and eigenvalues of a covariance (or correlation)
##matrix represent the “core” of a PCA: The eigenvectors (principal components)
##determine the directions of the new feature space,
##and the eigenvalues determine their magnitude.
##In other words, the eigenvalues explain the variance of the data along the new feature axes.

##Covariance Matrix
import numpy as np
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
print('Convariance matrix \n {}'.format(cov_mat))

##The more verbose way above was simply used for demonstration purposes,
##equivalently, we could have used the numpy cov function:
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

# compute eigendecomposition on the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvalues \n {}'.format(eig_vals))
print('\n Eigenvectors \n {}'.format(eig_vecs))

print('\n\n')

##Correlation Matrix
##Especially, in the field of “Finance,”
##the correlation matrix typically used instead of the covariance matrix.
##However, the eigendecomposition of the covariance matrix
##(if the input data was standardized)
##yields the same results as a eigendecomposition on the correlation matrix,
##since the correlation matrix can be understood as
##the normalized covariance matrix.

##Eigendecomposition of the standardized data based on the correlation matrix
cor_mat1 = np.corrcoef(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat1)


print('Eigenvalues \n {}'.format(eig_vals))
print('\n Eigenvectors \n {}'.format(eig_vecs))

print('\n\n')

##Eigendecomposition of the raw data based on the correlation matrix:
cor_mat2 = np.corrcoef(X.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

print('Eigenvalues \n {}'.format(eig_vals))
print('\n Eigenvectors \n {}'.format(eig_vecs))

print('\n\n')

##Singular Vector Decomposition
##
##While the eigendecomposition of the covariance or correlation matrix
##may be more intuitiuve, most PCA implementations perform a
##Singular Vector Decomposition (SVD) to improve the computational efficiency.
##So, let us perform an SVD to confirm that the result are indeed the same

u,s,v = np.linalg.svd(X_std.T)

print('u \n {}'.format(u))

##2 - Selecting Principal Components

##Sorting Eigenpairs
##
##The typical goal of a PCA is to reduce
##the dimensionality of the original feature space by projecting it
##onto a smaller subspace, where the eigenvectors will form the axes.
##However, the eigenvectors only define the directions of the new axis,
##since they have all the same unit length 1,
##which can confirmed by the following two lines of code

for ev in eig_vecs:
    np.testing.assert_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

##In order to decide which eigenvector(s) can dropped
##To do so, the common approach is to rank the eigenvalues
##from highest to lowest in order choose the top kk eigenvectors.

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


##Explained Variance
##After sorting the eigenpairs, the next question is “how many principal components are we going to choose for our new feature subspace?” A useful measure is the so-called “explained variance,” which can be calculated from the eigenvalues. The explained variance tells us how much information (variance) can be attributed to each of the principal components.

##tot = sum(eig_vals)
##var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
##cum_var_exp = np.cumsum(var_exp)
##
##with plt.style.context('seaborn-whitegrid'):
##    plt.figure(figsize=(6, 4))
##
##    plt.bar(range(4), var_exp, alpha=0.5, align='center',
##            label='individual explained variance')
##    plt.step(range(4), cum_var_exp, where='mid',
##             label='cumulative explained variance')
##    plt.ylabel('Explained variance ratio')
##    plt.xlabel('Principal components')
##    plt.legend(loc='best')
##    plt.tight_layout()
##    plt.show()

##The plot above clearly shows that most of the variance
##(72.77% of the variance to be precise) can be explained
##by the first principal component alone.
##The second principal component still bears some information (23.03%)
##while the third and fourth principal components can safely be
##dropped without losing to much information.
##Together, the first two principal components
##contain 95.8% of the information.

##Projection Matrix
##It’s about time to get to the really interesting part: The construction of the projection matrix that will be used to transform the Iris data onto the new feature subspace. Although, the name “projection matrix” has a nice ring to it, it is basically just a matrix of our concatenated top k eigenvectors.
##
##Here, we are reducing the 4-dimensional feature space to a 2-dimensional feature subspace, by choosing the “top 2” eigenvectors with the highest eigenvalues to construct our d×kd×k-dimensional eigenvector matrix W

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))


print('Matrix W:\n {}'.format(matrix_w))

##3 - Projection Onto the New Feature Space
##In this last step we will use the 4×2-dimensional projection matrix W
##to transform our samples onto the new subspace via the equation
##Y=X×W, where Y is a 150×2 matrix of our transformed samples

Y = X_std.dot(matrix_w)

##with plt.style.context('seaborn-whitegrid'):
##    plt.figure(figsize=(6, 4))
##    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
##                        ('blue', 'red', 'green')):
##        plt.scatter(Y[y==lab, 0],
##                    Y[y==lab, 1],
##                    label=lab,
##                    c=col)
##    plt.xlabel('Principal Component 1')
##    plt.ylabel('Principal Component 2')
##    plt.legend(loc='lower center')
##    plt.tight_layout()
##    plt.show()

##Shortcut - PCA in scikit-learn
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
        plt.scatter(Y_sklearn[y==lab, 0],
                    Y_sklearn[y==lab, 1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()
