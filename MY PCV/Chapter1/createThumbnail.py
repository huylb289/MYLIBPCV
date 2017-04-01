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
        inFile = fileImage
        outFile = os.path.splitext(inFile)[0] + "_thumbnail" + ".jpg"
        print("inFile {}".format(inFile))
        pil_im = IM.open(inFile)
        """
        The thumbnail() method takes a tuple
        specifying the new size and converts the image to a thumbnail image with size that fits
        with a tuple.
        To create a thumbnail with longest side 128 pixels

        this is not return any value
        """
        pil_im.thumbnail((128,128))
        pil_im.save(outFile)
        print("outFile {}".format(outFile))
        break # Comment out here
        
    return

if __name__ == '__main__':
    main()
                   
