from math import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from Linkage import Linkage
from Plate import Plate
from Neckbrace import Neckbrace


def main():
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(xmin=-10, xmax=10)
    ax.set_xlabel("X", fontsize=20)

    ax.set_ylim3d(ymin=-6, ymax=6)
    ax.set_ylabel("Y", fontsize=20)

    ax.set_zlim3d(zmin=0, zmax=8)
    ax.set_zlabel("Z", fontsize=20)

    # Neckbrace robot with ring radius of 5
    neckbrace = Neckbrace(5)
    neckbrace.plot()

    plt.show()


if __name__ == "__main__":
    main()