import typesetter
import numpy as np
from skimage import io, exposure
import graph

toolpath = typesetter.typeset('Hello, World!', 'Vera.ttf')

#zees = [[size_map[x] for x in path] for path in paths]

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#for path, z in zip(paths, zees):
#    x,y = zip(*path)
#    ax.plot(x,y,z)

#plt.show()
