import os
from MYLIBPCV.tools import sfm

exec(open("./load_vggdata.py").read())

# plotting the points in view 1
##plt.figure()
##plt.imshow(im1)
##plt.plot(points2D[0][0], points2D[0][1], '*')
##plt.axis('off')
##
##plt.figure()
##plt.imshow(im1)
##plt.plot(x[0],x[1],'r.')
##plt.axis('off')
##
##plt.show()

# plotting 3D points
##from mpl_toolkits.mplot3d import axes3d
##
##fig = plt.figure()
##ax = fig.gca(projection='3d')
##ax.plot(points3D[0],points3D[1],points3D[2],'k.')
##
##plt.show()

##from MYLIBPCV.tools import sfm
##
### index for points in first two views
##ndx = (corr[:,0] >= 0) & (corr[:,1] >= 0)
##
### get coordinates and make homogeneous
##x1 = points2D[0][:,corr[ndx,0]]
##x1 = np.vstack((x1, np.ones(x1.shape[1])))
##x2 = points2D[1][:,corr[ndx,1]]
##x2 = np.vstack((x2, np.ones(x2.shape[1])))
##
### compute F
##F = sfm.computeFundamental(x1,x2)
##
### compute the epipole
##e = sfm.computeEpipole(F)
##
### plotting
##plt.figure()
##plt.imshow(im1)
##
### plot each line individually, this gives nice colors
##for i in range(5):
##    sfm.plotEpipolarLine(im1, F, x2[:,i], e, False)
##plt.axis('off')
##
##plt.figure()
##plt.imshow(im2)
### plot each point individually, this gives same colors as the lines
##for i in range(5):
##    plt.plot(x2[0,i], x2[1,i], 'o')
##plt.axis('off')
##plt.show()

### Try triangulation on the Merton1
##from MYLIBPCV.tools import sfm
##
### index for points in first two views
##ndx = (corr[:,0] >= 0) & (corr[:,1] >= 0)
### get coordinates and make homogeneous
##x1 = points2D[0][:,corr[ndx,0]]
##x1 = np.vstack((x1, np.ones(x1.shape[1])))
##x2 = points2D[1][:,corr[ndx,1]]
##x2 = np.vstack((x2, np.ones(x2.shape[1])))
##
##Xtrue = points3D[:,ndx]
##Xtrue = np.vstack((Xtrue, np.ones(Xtrue.shape[1])))
##
### check first 3 points
##Xest = sfm.triangulate(x1, x2, P[0].P, P[1].P)
##print(Xest[:,:3])
##print(Xtrue[:,:3])
##
### plotting
##from mpl_toolkits.mplot3d import axes3d
##fig = plt.figure()
##ax = fig.gca(projection='3d')
##ax.plot(Xest[0],Xest[1],Xest[2],'ko')
##ax.plot(Xtrue[0],Xtrue[1],Xtrue[2],'r.')
##plt.axis('equal')
##plt.show()


# Estimate camera matrix
from MYLIBPCV.tools import sfm
from MYLIBPCV.tools import camera

corr = corr[:,0] # view 1
ndx3D = np.where(corr>=0)[0] # missing values are -1
ndx2D = corr[ndx3D]

# select visible points and make homogenous
x = points2D[0][:,ndx2D] # view 1
x = np.vstack((x, np.ones(x.shape[1])))
X = points3D[:, ndx3D]
X = np.vstack((X, np.ones(X.shape[1])))

# estimate P
Pest = camera.Camera(sfm.computeP(x,X))

# compare!
print(Pest.P/Pest.P[2,3])
print(P[0].P/P[0].P[2,3])

xest = Pest.project(X)

# plotting
plt.figure()
plt.imshow(im1)
plt.plot(x[0],x[1], 'bo')
plt.plot(xest[0], xest[1], 'r.')
plt.axis('off')

plt.show()
