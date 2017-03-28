"""
The Python Imaging Library(PIL) provides general image handling and lot of useful
basic image operations like resizing, cropping, rotating, color conversion...
"""

from PIL import Image as IM
from MYPCVLIB.tools import imtool	
# pre-check for dataFolder

dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

def main():
    imageFile = 'empire.jpg'
    inFile = os.path.join(dataFolder, imageFile)
    outFile = 'gray_empire.jpg'
    pil_im = IM.open(inFile)
    pil_im = pil_im.convert('L')
    pil_im.save(outFile)
    return


if __name__ == '__main__':
    main()
                     

