from PIL import Image as IM
from MYLIBPCV.tools import imtool
import os

dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

def main():
    # create thumbnail
    fileList = imtool.get_imlist(dataFolder)
    for fileImage in fileList:
        """
        resize() with a tuple giving the new size:
        """
        newSize = (128,128)
        rotateDegree = 45
        inFile = fileImage
        resizeFile = os.path.splitext(inFile)[0] + "_resized" + ".jpg"
        rotateFile = os.path.splitext(inFile)[0] + "_rotated" + ".jpg"
        print("inFile {}".format(inFile))
        pil_im = IM.open(inFile)
        resizeImage = pil_im.resize(newSize)
        rotateImage = pil_im.rotate(rotateDegree)
        resizeImage.save(resizeFile)
        rotateImage.save(rotateFile)
        print("resizeFile {}".format(resizeFile))
        print("rotateFile {}".format(rotateFile))
        break # Comment out here
        
    return

if __name__ == '__main__':
    main()
                   
