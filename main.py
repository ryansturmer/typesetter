import typesetter
import numpy as np
from skimage import io, exposure
import graph


def plot():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_autoscalez_on(False)
    ax.set_zlim([-500.0,0.0])
    toolpath.scale(z=-1.0)
    for path in toolpath:
        x,y,z = zip(*path)
        ax.plot(x,y,z)

    plt.show()

if __name__ == "__main__":
    toolpath = typesetter.typeset('Hello, World!', 'Vera.ttf')
    plot()
