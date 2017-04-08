import pickle
from MYLIBPCV.tools import knn, imtools
import numpy as np
import matplotlib.pyplot as plt

# load 2D points using Pickle
with open('points_normal.pkl', 'rb') as f:
    class1 = pickle.load(f)
    class2 = pickle.load(f)
    labels = pickle.load(f)

model = knn.KnnClassifier(labels, np.vstack((class1, class2)))

# load test data using Pickle
with open('points_ring.pkl', 'rb') as f:
    class1 = pickle.load(f)
    class2 = pickle.load(f)
    labels = pickle.load(f)
# test on the first point
print (model.classify(class1[0]))

def classify(x,y,model=model):
    return np.array([model.classify([xx,yy]) for (xx,yy) in zip(x,y)])

# plot the classification boundary
imtools.plot_2D_boundary([-6, 6, -6, 6],[class1, class2], classify, [1,-1])
plt.show()
