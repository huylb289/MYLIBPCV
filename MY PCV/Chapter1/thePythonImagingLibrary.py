"""
The Python Imaging Library(PIL) provides general image handling and lot of useful
basic image operations like resizing, cropping, rotating, color conversion...
"""

from PIL import Image as IM
from MYLIBPCV.tools import imtool
import os
# pre-check for dataFolder

dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

def main():
    # convert all the image from dataFolder into gray color
    fileList = imtool.get_imlist(dataFolder)
    for fileImage in fileList:
        inFile = fileImage
        outFile = os.path.splitext(inFile)[0] + "_gray" + ".jpg"
        print("inFile {}".format(inFile))
        pil_im = IM.open(inFile)
        pil_im = pil_im.convert('L')
        pil_im.save(outFile)
        print("outFile {}".format(outFile))
        break # Comment out here
        
    return

if __name__ == '__main__':
    main()
                     

