from MYLIBPCV.tools import imtools, pca
import pickle
from scipy.cluster import vq
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

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
centroids, distortion = vq.kmeans(projected, 4)

code, distance = vq.vq(projected, centroids)

##Same as before, code contains the cluster assignment for each image. In this case, we
##tried k = 4.
##We also chose to “whiten” the data using SciPy ’s whiten() , normalizing so
##that each feature has unit variance.

# plot clusters
for k in range(4):
    ind = np.where(code==k)[0]
    plt.figure()
    plt.gray()
    for i in range(np.minimum(len(ind), 40)):
        plt.subplot(4, 10, i+1)
        plt.imshow(immatrix[ind[i]].reshape((25,25)))
        plt.axis('off')

plt.show()
##
### height and width
##h,w = 1200,1200
##
### create a new image with a white background
##img = Image.new('RGB',(w,h),(255,255,255))
##draw = ImageDraw.Draw(img)
##
### draw axis
##draw.line((0,h/2,w,h/2),fill=(255,0,0))
##draw.line((w/2,0,w/2,h),fill=(255,0,0))
##
### scale coordinates to fit
##scale = np.absolute(projected).max(0)
##scaled = np.floor(np.array([ (p / scale) * (w/2-20,h/2-20) + \
##                             (w/2,h/2) for p in projected]))
##
### paste thumbnail of each image
##for i in range(imnbr):
##    nodeim = Image.open(imlist[i])
##    nodeim.thumbnail((25,25))
##    ns = nodeim.size
##    img.paste(nodeim,(scaled[i][0]-ns[0]//2,scaled[i][1]- \
##                      ns[1]//2,scaled[i][0]+ns[0]//2+1,scaled[i][1]+ns[1]//2+1))
##img.save('pca_font.jpg')
