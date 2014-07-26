import numpy as np

def neighbors(x,y):
    return (x-1,y-1),(x-1,y), (x-1,y+1), (x,y-1), (x,y+1), (x+1, y-1), (x+1, y), (x+1, y+1)

def create_from_image_skeleton(matrix):
    '''
    Input: matrix - A 2D array that represents a skeletonized image.
    Output: A graph dictionary which maps coordinates to pixel values.  The coordinates are the x,y locations of
            nonzero pixels in the input matrix, and the pixel values are intensities that correspond to the values of 
            those pixels.  Adjacency is determined by pixels in the 3x3 area surrounding the pixel in question
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
    Output: A list of graph dictionary, one for each disjoint subgraph in the input
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


def create_path_from_graph(graph):
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
    return path
