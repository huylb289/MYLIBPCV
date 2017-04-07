import numpy as np 
from MYLIBPCV.tools import hcluster

class1 = 1.5 * np.random.randn(100,2)
class2 = np.random.randn(100,2) + np.array([5,5])
features = np.vstack((class1,class2))

tree = hcluster.hcluster(features)
clusters = tree.extractClusters(5)

print('number of cluster {}'.format(len(clusters)))

for c in clusters:
    print (c.getClusterElements())
