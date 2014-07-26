import skimage.morphology
from freetype import *
import numpy as np
from skimage import morphology, data, io, img_as_bool, img_as_float, exposure, filter as skfilter
import itertools
import numpy

IMAGE_FILENAME = 'images/text.png'
SKELETON_FILENAME = 'images/skel.png'
MAXIS_FILENAME = 'images/maxis.png'
PRETTY_FILENAME = 'images/pretty.png'

TYPESETTER_POINT_SIZE = 1000

'''
Data Structures Use in File:

graph - A graph maps coordinates (x,y) to a list of neighboring coordinates.  Used to create toolpaths.
size_map - A size map maps coordinates (x,y) to a pixel size (this is part of the output of the medial axis transform)
matrix - A 2D image

'''

def typeset(text, font, point_size=1000):

    # Generate an image skeleton
    matrix = get_text_image(text, font, 1000)

    # Save off a copy for inspection
    io.imsave(PRETTY_FILENAME, exposure.rescale_intensity(matrix, out_range=(0,255)).astype(np.uint8))
    
    # Generate graphs
    graphs, size_map = create_from_image_skeleton(matrix)

    # Turn graphs into toolpath information
    path = MultiToolpath([create_path_from_graph(g, size_map) for g in graphs])

    return path

class Toolpath(object):
    def __init__(self):
        self.points = []
    def append(self, p):
        self.points.append(p)
    def __len__(self):
        return len(self.points)
    def __iter__(self):
        return iter(self.points)
    def get_extents(self):
        "The bounds of the toolpath, in the form: (xmin,xmax),(ymin,ymax),(zmin,zmax)"
        x,y,z = zip(*self.points)
        return (min(x),max(x)),(min(y),max(y)),(min(z),max(z))
    def scale(self, x=1.0,y=1.0,z=1.0):
        for i, (xpt,ypt,zpt) in enumerate(self.points):
            self.points[i] = x*float(xpt), y*float(ypt), z*float(zpt)
    def translate(self, x=0, y=0, z=0):
        for i, (xpt, ypt, zpt) in enumerate(self.points):
            self.points[i] = xpt + x, ypt + y, zpt + z

class MultiToolpath(object):
    def __init__(self, paths=None):
        if paths:
            self.paths = paths[:]
    def append(self, path):
        self.paths.append(path)
    def __len__(self):
        return len(self.paths)
    def __iter__(self):
        return iter(self.paths)
    @property
    def all_points(self):
        return itertools.chain(*self)
    def get_extents(self):
        extents = [path.get_extents() for path in self.paths]
        x_extents, y_extents, z_extents = zip(*extents)
        minmax = lambda a,b : (min((a[0],b[0])), max((a[1],b[1])))
        return reduce(minmax, x_extents), reduce(minmax, y_extents), reduce(minmax, z_extents)
    def scale(self, x=1.0,y=1.0,z=1.0):
        for path in self.paths:
            path.scale(x,y,z)
    def translate(self, x=0,y=0,z=0):
        for path in self.paths:
            path.translate(x,y,z)
    def get_dimensions(self):
        (xmin,xmax),(ymin,ymax),(zmin,zmax) = self.get_extents()
        return (xmax-xmin),(ymax-ymin),(zmax-zmin)
        
def POINT_SIZE(x):
    return x*64

def get_text_extents(text, filename, size=500):
    face = Face(filename)
    face.set_char_size(POINT_SIZE(size))
    flags = FT_LOAD_RENDER
    pen = FT_Vector(0,0)
    previous = None # 0
    xmin, xmax = 0, 0
    ymin, ymax = 0, 0
    for c in text:
        face.load_char(c, flags)
        kerning = face.get_kerning(previous, c)
        previous = c
        bitmap = face.glyph.bitmap
        pitch  = face.glyph.bitmap.pitch
        width  = face.glyph.bitmap.width
        rows   = face.glyph.bitmap.rows
        top    = face.glyph.bitmap_top
        left   = face.glyph.bitmap_left
        pen.x += kerning.x
        x0 = (pen.x >> 6) + left
        x1 = x0 + width
        y0 = (pen.y >> 6) - (rows - top)
        y1 = y0 + rows
        xmin, xmax = min(xmin, x0),  max(xmax, x1)
        ymin, ymax = min(ymin, y0), max(ymax, y1)
        pen.x += face.glyph.advance.x
        pen.y += face.glyph.advance.y
    return xmin, xmax, ymin, ymax

def get_text(text, filename, size):
    face = Face(filename)
    face.set_char_size(POINT_SIZE(size))
    pen = FT_Vector(0,0)
    flags = FT_LOAD_RENDER
    xmin,xmax,ymin,ymax = get_text_extents(text, filename, size)
    L = np.zeros((ymax-ymin, xmax-xmin),dtype=np.ubyte)
    previous = 0
    pen.x, pen.y = 0, 0
    for c in text:
        import sys
        sys.stdout.flush()
        face.load_char(c, flags)
        kerning = face.get_kerning(previous, c)
        previous = c
        bitmap = face.glyph.bitmap
        pitch  = face.glyph.bitmap.pitch
        width  = face.glyph.bitmap.width
        rows   = face.glyph.bitmap.rows
        top    = face.glyph.bitmap_top
        left   = face.glyph.bitmap_left
        pen.x += kerning.x
        x = (pen.x >> 6) - xmin + left
        y = (pen.y >> 6) - ymin - (rows - top)
        data = []
        d = bitmap.buffer[:]
        for j in range(rows):
            data.append(d[j*pitch:j*pitch+width])
        data = list(itertools.chain(data))
        if len(data):
            Z = np.array(data,dtype=np.ubyte).reshape(rows, width)
            L[y:y+rows,x:x+width] |= Z[::-1,::1]
        pen.x += face.glyph.advance.x
        pen.y += face.glyph.advance.y

    # Create a 10 pixel border for the image    
    L = np.flipud(L)
    rows, cols = L.shape
    row = 10*[[0]*cols]
    L = np.vstack((row, L, row))

    rows, cols = L.shape
    col = np.array([[0]*10]*rows)
#    print col.shape
    L = np.hstack((col, L, col))
    return L

def get_text_image(text, font, point_size):
    text = get_text(text, font, point_size)
    io.imsave(IMAGE_FILENAME, text)
    thresh = skfilter.threshold_otsu(text)
    binary = text > thresh
    skel, distance = morphology.medial_axis(binary, return_distance=True)
    distance = distance.astype(np.uint16)
    skel = skel.astype(np.uint16)
    return skel*distance

def neighbors(x,y):
    return (x-1,y-1),(x-1,y), (x-1,y+1), (x,y-1), (x,y+1), (x+1, y-1), (x+1, y), (x+1, y+1)

def create_from_image_skeleton(matrix):
    '''
    Input: matrix - A 2D array that represents a skeletonized image.
    Output: a list of graphs, and a size map
            graphs : map coordinates (x,y) to a list of neighboring coordinates.  Neighbors are adjacent (3x3) pixels that are nonzero
            size_map : map coordinates (x,y) to pixel intensities for all nonzero pixels
    '''
    points = np.argwhere(matrix)
    size_map = {}
    for i,j in points:
        size_map[(i,j)] = matrix[i][j]

    graph = {}
    for i,j in size_map:
        l = []
        graph[(i,j)] = l
        for neighbor in neighbors(i,j):
            if neighbor in size_map:
                l.append(neighbor)

    return split(graph), size_map

def split(graph):
    '''
    Input: graph - A graph dictionary
    Output: A list of graph dictionaries, one for each disjoint subgraph in the input
    '''
    # We label the graph first, then split it
    # Labeled graph maps coordinate -> label
    labeled_graph = {}

    # N is the graph label.  Label graphs from 0 to the total number of graphs minus one.
    N = 0

    # We have to check every node in the parent graph to make sure we catch all disconnected subgraphs
    for node in graph:
        # Don't start a search on areas we've already labeled
        if node in labeled_graph:
            continue

        # Start a new search at this node
        visit_stack = [node]
        while True:
            try:
                node = visit_stack.pop()
            except:
                break # Empty visit stack means done labeling

            # Apply label
            labeled_graph[node] = N

            # Visit neighbors if they are unlabeled
            neighbors = graph[node]
            visitable_neighbors = filter(lambda n : n not in labeled_graph, neighbors)
            visit_stack.extend(visitable_neighbors)
        
        # Done with this subgraph, create new label
        N += 1

    # Turn the labeled graph into N separate subgraphs
    retval = [dict() for i in range(N)]
    for k, v in labeled_graph.items():
        retval[v][k] = graph[k]

    return retval

def create_path_from_graph(graph, size_map):
    visited = {}
    visit_stack = []
    path = []
    node = graph.keys()[0]
    visit_stack.append(node)
    while True:
        try:
            node = visit_stack.pop()
        except:
            break 
        path.append(node)
        visited[node] = visited.get(node,0) + 1
        if len(visited) == len(graph):
            break
        else:
            # ranked_neighbors is list of the form: (rank, neighbor) where rank is the number of visits a node has seen
            neighbors = graph[node]
            ranked_neighbors = [(visited.get(neighbor, 0), neighbor) for neighbor in neighbors]
            ranked_neighbors.sort(reverse=True)
            for rank, neighbor in ranked_neighbors:
                visit_stack.append(neighbor)
    retval = Toolpath()
    for x,y in path:
        retval.append((x,y, size_map[(x,y)]))
    return retval
