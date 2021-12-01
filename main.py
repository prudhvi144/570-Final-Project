from math import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from Linkage import Linkage
from Plate import Plate
import Neckbrace as neckbrace


def main():
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(xmin=-10, xmax=10)
    ax.set_xlabel("X", fontsize=20)

    ax.set_ylim3d(ymin=-6, ymax=6)
    ax.set_ylabel("Y", fontsize=20)

    ax.set_zlim3d(zmin=0, zmax=8)
    ax.set_zlabel("Z", fontsize=20)

    my_plate = Plate()
    second_plate = Plate()
    second_plate.vertices[2, :] = 4 * [4.5]
    my_plate.plot('b')
    second_plate.plot('g')

    first_link = Linkage(my_plate.radius, pi / 6)
    first_link.plot('r')

    second_link = Linkage(my_plate.radius, 5 * pi / 6)
    second_link.plot('r')

    third_link = Linkage(my_plate.radius, 7 * pi / 6)
    third_link.plot('r')

    fourth_link = Linkage(my_plate.radius, 11 * pi / 6)
    fourth_link.plot('r')

    plt.show()


if __name__ == "__main__":
    main()