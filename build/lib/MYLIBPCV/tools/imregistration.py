from xml.dom import minidom
import numpy as np
from scipy import linalg
from scipy import ndimage
from scipy.misc import imsave
import os
from PIL import Image as IM
import matplotlib.pyplot as plt


def readPointsFromXML(xmlFileName):
    """ Reads control points for face alignment"""

    xmldoc = minidom.parse(xmlFileName)
    facelist = xmldoc.getElementsByTagName('face')
    faces = {}

    for xmlFace in facelist:
        fileName = xmlFace.attributes['file'].value
        xf = int(xmlFace.attributes['xf'].value)
        yf = int(xmlFace.attributes['yf'].value)
        xs = int(xmlFace.attributes['xs'].value)
        ys = int(xmlFace.attributes['ys'].value)
        xm = int(xmlFace.attributes['xm'].value)
        ym = int(xmlFace.attributes['ym'].value)
        
        faces[fileName] = np.array([xf,yf,xs,ys,xm,ym])

    return faces

def computeRigidTransform(refpoints, points):
    """ Computes rotation, scale and translation for
    aligning points to refpoints."""

    A = np.array([  [points[0], -points[1], 1, 0],
                    [points[1],  points[0], 0, 1],
                    [points[2], -points[3], 1, 0],
                    [points[3],  points[2], 0, 1],
                    [points[4], -points[5], 1, 0],
                    [points[5],  points[4], 0, 1]])

    y = np.array([ refpoints[0],
                   refpoints[1],
                   refpoints[2],
                   refpoints[3],
                   refpoints[4],
                   refpoints[5]])

    # least sq solution to minimize ||Ax - y||
    a,b,tx,ty = linalg.lstsq(A,y)[0]
    R = np.array([[a, -b], [b, a]]) # rotation matrix incl scale

    return R, tx, ty

def rigidAlignment(faces, path, plotflag=False):
    """ Align images rigidly and save as new images.
    path determines where the aligned images are saved set plotflag=True to plot the images"""

    # take the points in the first image as reference points
    refpoints = list(faces.values())[0]

    # warp each image using affine transform
    for face in faces:
        points = faces[face]
        R, tx, ty = computeRigidTransform(refpoints, points)
        T = np.array([[R[1][1], R[1][0]], [R[0][1], R[0][0]]])

        im = np.array(IM.open(os.path.join(path,face)))
        im2 = np.zeros(im.shape, 'uint8')

        # warp each color channel
        for i in range(len(im.shape)):
            im2[:,:,i] = ndimage.affine_transform(im[:,:,i], linalg.inv(T), \
                                                  offset=[-ty,-tx])

        if plotflag:
            plt.imshow(im2)
            plt.show()

        # crop away border and save aligned images
        h,w = im2.shape[:2]
        border = (w+h)/20
        border = int(border)

        # crop away border
        imsave(os.path.join(path, 'aligned/' + face), im2[border:int(h-border), border:int(w-border), :])

