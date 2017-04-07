from scipy.cluster import vq
import numpy as np
import matplotlib.pyplot as plt

class1 = 1.5 * np.random.randn(100,2)
class2 = np.random.randn(100,2) + np.array([5,5])
features = np.vstack((class1, class2))

centroids, variance = vq.kmeans(features, k_or_guess=2)
code, distance = vq.vq(features, centroids)


# Data before clustering
plt.figure()
ndx = np.where(code==0)[0]
plt.plot(class1[:,0], class1[:,1], '*')
plt.plot(class2[:,0], class2[:,1], '*')
ndx = np.where(code==1)[0]
plt.axis('off')
plt.show()


plt.figure()
ndx = np.where(code==0)[0]
plt.plot(features[ndx,0], features[ndx,1], '*')
ndx = np.where(code==1)[0]
plt.plot(features[ndx,0], features[ndx,1], 'r.')
plt.plot(centroids[:,0], centroids[:,1], 'go')
plt.axis('off')
plt.show()
