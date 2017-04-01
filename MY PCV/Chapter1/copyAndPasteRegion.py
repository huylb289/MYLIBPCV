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
        The region is defined by a 4-tuple, where coordinates are (left, upper, right, lower)
        """
        box = (100, 100, 400, 400)
        inFile = fileImage
        outFile = os.path.splitext(inFile)[0] + "_copyAndPasteRegion" + ".jpg"
        print("inFile {}".format(inFile))
        pil_im = IM.open(inFile)
        region = pil_im.crop(box) # crop the region
        region = region.transpose(IM.ROTATE_180) # rotate the region
        pil_im.paste(region, box)
        pil_im.save(outFile)
        print("outFile {}".format(outFile))
        break # Comment out here
        
    return

if __name__ == '__main__':
    main()
                   
