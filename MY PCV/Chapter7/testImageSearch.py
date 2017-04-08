




src = imagesearch.Searcher('test.db')
locs,descr = sift.read_features_from_file(featlist[0])
iw = voc.project(descr)
print 'ask using a histogram...'
print src.candidates_from_histogram(iw)[:10]


src = imagesearch.Searcher('test.db')
print 'try a query...'
print src.query(imlist[0])[:10]
