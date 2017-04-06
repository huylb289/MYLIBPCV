from MYLIBPCV.tools import imregistration
from MYLIBPCV.tools import imtools
import os
import numpy as np

# read image to array
dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

# load the location of control points
xmlFileName = os.path.join(dataFolder, 'jkfaces/jkfaces.xml')
points = imregistration.readPointsFromXML(xmlFileName)

# register
alignmentFolder = os.path.join(dataFolder, 'jkfaces')
imregistration.rigidAlignment(points, alignmentFolder)

alignedFolder = os.path.join(dataFolder, 'jkfaces/aligned')
imlist = imtools.getImlist(alignedFolder)


# where mask is a binary image of the same size
# mask: ellipse mask binary, defining the area of face
##mask = ellipse binary shape(230x230)
##immatrix = np.array([mask*array(Image.open(imlist[i]).convert('L')).flatten() for i in range(150)],'f')
