from itertools import combinations
import numpy as np
from PIL import Image, ImageDraw

class ClusterNode(object):
    def __init__(self, vec, left, right, distance=0.0, count=1):
        self.left = left
        self.right = right
        self.vec = vec
        self.distance = distance
        self.count = count

    def extractClusters(self,dist):
        """ Extract list of sub-tree clusters from
        hcluster tree with distance<dist"""

        if self.distance < dist:
            return [self]
        return self.left.extractClusters(dist) + \
               self.right.extractClusters(dist)

    def getClusterElements(self):
        """ Return ids for elements in a cluster sub-tree"""
        return self.left.getClusterElements() + self.right.getClusterElements()

    def getHeight(self):
        """ Return the height of a node, height is sum of each branch"""
        return self.left.getHeight() + self.left.getHeight()

    def getDepth(self):
        """ Return the depth of a node, depth is
        max of each child plus own distance. """
        
        return max(self.left.getDepth(), self.right.getDepth()) + self.distance

    def draw(self,draw,x,y,s,imlist,im):
        """ Draw nodes recursively with image
        thumbnails for leaf nodes. """
        
        h1 = np.int(self.left.getHeight()*20 / 2)
        h2 = np.int(self.right.getHeight()*20 /2)
        top = y-(h1+h2)
        bottom = y+(h1+h2)
        
        # vertical line to children
        draw.line((x,top+h1,x,bottom-h2),fill=(0,0,0))
        
        # horizontal lines
        ll = self.distance*s
        draw.line((x,top+h1,x+ll,top+h1),fill=(0,0,0))
        draw.line((x,bottom-h2,x+ll,bottom-h2),fill=(0,0,0))
        
        # draw left and right child nodes recursively
        self.left.draw(draw,x+ll,top+h1,s,imlist,im)
        self.right.draw(draw,x+ll,bottom-h2,s,imlist,im)

class ClusterLeafNode(object):
    def __init__(self,vec,id):
        self.vec = vec
        self.id = id
        
    def extractClusters(self,dist):
        return [self]
    
    def getClusterElements(self):
        return [self.id]
    
    def getHeight(self):
        return 1

    def getDepth(self):
        return 0

    def draw(self,draw,x,y,s,imlist,im):
        nodeim = Image.open(imlist[self.id])
        nodeim.thumbnail([20,20])
        ns = nodeim.size
        im.paste(nodeim,[int(x),int(y-ns[1]//2),int(x+ns[0]),int(y+ns[1]-ns[1]//2)])

def L2dist(v1,v2):
    return np.sqrt(np.sum((v1-v2)**2))

def L1dist(v1,v2):
    return np.sum(np.abs(v1-v2))

def hcluster(features,distfcn=L2dist):
    """ Cluster the rows of features using
    hierarchical clustering. """
    
    # cache of distance calculations
    distances = {}
    
    # initialize with each row as a cluster
    node = [ClusterLeafNode(np.array(f),id=i) for i,f in enumerate(features)]
    
    while len(node)>1:
        closest = float('Inf')
        
        # loop through every pair looking for the smallest distance
        for ni,nj in combinations(node,2):
            if (ni,nj) not in distances:
                # distfcn=L2dist
                distances[ni,nj] = distfcn(ni.vec,nj.vec)
            d = distances[ni,nj]
            
            if d<closest:
                closest = d
                lowestpair = (ni,nj)
        ni,nj = lowestpair
        
        # average the two clusters
        new_vec = (ni.vec + nj.vec) / 2.0
        
        # create new node
        new_node = ClusterNode(new_vec,left=ni,right=nj,distance=closest)
        node.remove(ni)
        node.remove(nj)
        node.append(new_node)
    return node[0]

def drawDendrogram(node,imlist,filename='clusters.jpg'):
    """ Draw a cluster dendrogram and save to a file. """
    
    # height and width
    rows = node.getHeight()*20
    cols = 1200
    # scale factor for distances to fit image width
    s = np.float(cols-150)/node.getDepth()
    # create image and draw object
    im = Image.new('RGB',(cols,rows),(255,255,255))

    draw = ImageDraw.Draw(im)

    # initial line for start of tree
    draw.line((0, rows/2, 20, rows/2), fill=(0,0,0))

    node.draw(draw,20,(rows/2),s,imlist,im)
    im.save(filename)
    im.show()
