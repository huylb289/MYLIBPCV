from MYLIBPCV.tools import imtools, pca
import pickle
from scipy.cluster import vq
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

##Spectral clustering methods are an interesting type of clustering algorithm that have a
##different approach compared to k-means and hierarchical clustering.

##A similarity matrix (or affinity matrix, or sometimes distance matrix) for n elements (for
##example images) is an n × n matrix with pair-wise similarity scores.
##It gets its name from the use of the spectrum of a matrix constructed from a similarity
##matrix. The eigenvectors of this matrix are used for dimensionality reduction and then
##clustering.

##n x n similarity matrix S, called the Laplacian matrix
##Sometimes L = D −1/2 SD −1/2 is used as the Laplacian matrix instead, but the choice doesn’t really matter
##since it only changes the eigenvalues, not the eigenvectors.



##http://docs.scipy.org/doc/scipy/reference/cluster.vq.html

dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

imlistPath = os.path.join(dataFolder, 'a_selected_thumbs')
imlist = imtools.getImlist(imlistPath)
imnbr = len(imlist)

# load model file
with open('../Chapter1/font_pca_models.pkl', 'rb') as f:
    immean = pickle.load(f)
    V = pickle.load(f)


# create matrix to store all flattened images
immatrix = np.array([np.array(Image.open(im)).flatten()
                    for im in imlist],'f')

# Add new Here
# perform PCA
##V,S,immean = pca.pca(immatrix)

# project on the 40 first PCs
immean = immean.flatten()
projected = np.array([np.dot(V[:40],immatrix[i]-immean) for i in range(imnbr)])

# k-means
projected = vq.whiten(projected)

n = len(projected)

# compute distance matrix
S = np.array([[np.sqrt(np.sum((projected[i]-projected[j])**2))\
               for i in range(n)] for j in range(n)], 'f')

# create Laplacian matrix
rowsum = np.sum(S, axis=0)
D = np.diag(1/np.sqrt(rowsum))
I = np.identity(n)
L = I - np.dot(D, np.dot(S,D))

# compute eigenvectors of L
U, sigma, V = np.linalg.svd(L)

k = 5
# create feature vector from k first eigenvectors
# by stacking eigenvectors as columns
features = np.array(V[:k]).T

# Normalize a group of observations on a per feature basis.
# k-means
features = vq.whiten(features)
centroids, distortion = vq.kmeans(features, k)
code, distance = vq.vq(features, centroids)

for c in range(k):
    ind = np.where(code==c)[0]
    plt.figure()
    for i in range(np.minimum(len(ind), 39)):
        im = Image.open(imlist[ind[i]])
        plt.gray()
        plt.subplot(4, 10, i+1)
        plt.imshow(np.array(im))
        plt.axis('equal')
        plt.axis('off')

plt.show()
