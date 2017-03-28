import camera
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load points
points = np.loadtxt('./Model House 3D/house.p3d').T
points = np.vstack((points, np.ones(points.shape[1]))) # attach 1 rows of ones on the last row

# setup camera
attach_array = np.array([[0],
                         [0],
                         [-10]])
P = np.hstack((np.eye(3), attach_array))
cam = camera.Camera(P)
x = cam.project(points)

# plot 3d
##fig = plt.figure()
##ax = Axes3D(fig)
##ax.scatter(x[0], x[1], x[2])
##plt.show()

# plot projection
plt.figure()
plt.plot(x[0],x[1], 'k.')


# create transformation
r = 0.05*np.random.rand(3)
rot = camera.Camera.rotation_matrix(r)

# rotate camera and project
plt.figure()
for t in range(20):
    cam.P = np.dot(cam.P, rot)
    x = cam.project(points)
    plt.plot(x[0], x[1], 'k.')

plt.show()


