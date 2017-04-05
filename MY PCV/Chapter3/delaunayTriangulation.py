import matplotlib.delaunay as md
import numpy as np
import matplotlib.pyplot as plt

##Warning (from warnings module):
##  File "/usr/lib/python3/dist-packages/matplotlib/cbook.py", line 137
##    warnings.warn(message, mplDeprecation, stacklevel=1)
##MatplotlibDeprecationWarning: The matplotlib.delaunay module was deprecated in version 1.4. Use matplotlib.tri.Triangulation instead.

x, y = np.array(np.random.standard_normal((2,100)))
centers, edges, tri, neighbors = md.delaunay(x,y)

plt.figure()
for t in tri:
    tExt = [t[0], t[1], t[2], t[0]] # add first point to end
    plt.plot(x[tExt], y[tExt], 'r')

plt.plot(x,y, '*')
plt.axis('off')
plt.show()
