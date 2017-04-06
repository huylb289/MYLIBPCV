from MYLIBPCV.tools import stereo
from PIL import Image as IM
import numpy as np
import matplotlib.pyplot as plt
import os

dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

imlPath = os.path.join(dataFolder, 'tsukuba', 'scene1.row3.col3.ppm')
iml = np.array(IM.open(imlPath).convert('L'),'f')
imrPath = os.path.join(dataFolder, 'tsukuba', 'scene1.row3.col4.ppm')
imr = np.array(IM.open(imrPath).convert('L'),'f')

# starting displacement and steps
steps = 12
start = 4
# width for ncc
wid = 9
res = stereo.planeSweepNCC(iml,imr,start,steps,wid)

import scipy.misc
scipy.misc.imsave('depth.png',res)

res = stereo.planeSweepGauss(iml,imr,start,steps,wid)

import scipy.misc
scipy.misc.imsave('depth_gaussian.png',res)
