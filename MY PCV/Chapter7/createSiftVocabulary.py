import numpy as np
from PIL import Image as IM
import matplotlib.pyplot as plt
from MYLIBPCV.tools import imtools, sift
import os

dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

imlistPath = os.path.join(dataFolder, 'ukbench/dataTrain')
imlist = imtools.getImlist(imlistPath)

nbrImages = len(imlist)
featlist = [imlist[i][:-3]+'sift' for i in range(nbrImages)]

for i in range(nbrImages):
    sift.processImage(imlist[i], featlist[i])    
