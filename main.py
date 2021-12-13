from math import pi
from operator import pos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from Linkage import Linkage
from Plate import Plate
from Neckbrace import Neckbrace


def main():
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(xmin=-7, xmax=7)
    ax.set_xlabel("X", fontsize=20)

    ax.set_ylim3d(ymin=-7, ymax=7)
    ax.set_ylabel("Y", fontsize=20)

    ax.set_zlim3d(zmin=0, zmax=14)
    ax.set_zlabel("Z", fontsize=20)

    # Neckbrace robot with ring radius of 5
    neckbrace = Neckbrace(5, 4.5,
                          [pi / 6, 5 * pi / 6, 7 * pi / 6, 11 * pi / 6])
    # neckbrace.plot(np.vstack((pi / 16, pi / 16)))
    # plt.show()

    div_pi_by = 18
    step_1 = np.linspace(pi / div_pi_by, 0, 5, endpoint=False)
    step_2 = np.linspace(0, -pi / div_pi_by, 5, endpoint=False)
    step_3 = np.linspace(-pi / div_pi_by, 0, 5, endpoint=False)
    step_4 = np.linspace(0, pi / div_pi_by, 5)

    positions = np.hstack((step_1, step_2, step_3, step_4))
    positions = np.vstack((positions, np.roll(positions, 5)))

    azimuth = -84
    elevation = 6
    neckbrace.animate(positions, "normal_first.gif", azimuth, elevation)


if __name__ == "__main__":
    main()