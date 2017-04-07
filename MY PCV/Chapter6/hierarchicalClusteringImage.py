import numpy as np 
from MYLIBPCV.tools import hcluster, imtools
from PIL import Image as IM
import os
import matplotlib.pyplot as plt

dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

imlistPath = os.path.join(dataFolder, 'sunsets', 'flickr-sunsets-small')

# create a list of images
imlist = imtools.getImlist(imlistPath)

# extract feature vector (8 bins per color channel)
features = np.zeros([len(imlist), 512])

for i,f in enumerate(imlist):
    im = np.array(IM.open(f))
    
    # multi-dimensional histogram
    h,edges = np.histogramdd(im.reshape(-1,3),8,normed=True,\
                          range=[(0,255),(0,255),(0,255)])
    
    features[i] = h.flatten()
    
tree = hcluster.hcluster(features)

hcluster.drawDendrogram(tree,imlist,filename='sunset.pdf')

# visualize clusters with some (arbitrary) threshold
##clusters = tree.extractClusters(0.23*tree.distance)

### plot images for clusters with more than 3 elements
##for c in clusters:
##    elements = c.getClusterElements()
##    nbr_elements = len(elements)
##    if nbr_elements>3:
##        plt.figure()
##        for p in range(np.minimum(nbr_elements,20)):
##            plt.subplot(4,5,p+1)
##            im = np.array(IM.open(imlist[elements[p]]))
##            plt.imshow(im)
##    plt.axis('off')
##plt.show()
