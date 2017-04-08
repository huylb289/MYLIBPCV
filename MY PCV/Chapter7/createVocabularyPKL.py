import pickle
from MYLIBPCV.tools import sift,imagesearch, vocabulary



dataFolder = '../../data/'
if not os.path.exists(dataFolder):
    print('Data Folder not exist')
    exit()

imlistPath = os.path.join(dataFolder, 'ukbench/dataTrain')
imlist = imtools.getImlist(imlistPath)

nbrImages = len(imlist)
featlist = [imlist[i][:-3]+'sift' for i in range(nbrImages)]

voc = vocabulary.Vocabulary('ukbenchtest')
voc.train(featlist,1000,10)
# saving vocabulary
with open('vocabulary.pkl', 'wb') as f:
pickle.dump(voc,f)
print ('vocabulary is:', voc.name, voc.nbr_words)





# load vocabulary
with open('vocabulary.pkl', 'rb') as f:
    voc = pickle.load(f)
    
# create indexer
indx = imagesearch.Indexer('test.db',voc)
indx.create_tables()

# go through all images, project features on vocabulary and insert
for i in range(nbrImages)[:100]:
    locs,descr = sift.readFeaturesFromFile(featlist[i])
    indx.add_to_index(imlist[i],descr)

# commit to database
indx.db_commit()
