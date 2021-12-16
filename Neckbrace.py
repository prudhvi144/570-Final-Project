from math import pi
from os import remove
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import imageio
from numpy.lib.function_base import angle
from Plate import Plate
from Linkage import Linkage
from Polygon import Polygon


class Neckbrace:
    def __init__(self,
                 plate_radius: float,
                 height: float,
                 link_angles=List[float]) -> None:

        self._verify(plate_radius, height, link_angles)

        self.plate_radius = plate_radius
        self.height = height
        self.link_angles = link_angles

        flexible_bodies = [Plate(self.plate_radius, self.height)]
        flexible_bodies.extend([
            Linkage(self.plate_radius, self.height, angle)
            for angle in self.link_angles
        ])

        # .extend(
        # [Linkage(self.plate_radius, angle) for angle in self.link_angles])

        self.bodies = {
            "rigid": Plate(self.plate_radius, 0),
            "flexible": flexible_bodies
        }

        self.potential = None

    def _verify(self,
                plate_radius: float,
                height: float,
                link_angles=List[float]) -> None:
        assert plate_radius > 0.0, 'Neckbrace radius must be greater than 0.0'
        assert height > 0.0, 'Neckbrace height must be greater than 0.0'
        assert len(link_angles) > 0, 'Number of links must be greater than 0'

    def animate(self, angles: np.ndarray, output_file: str, azimuth: float,
                elevation: float,x,y,z) -> None:
        filenames = []
        frame = 0
        for angle in angles.T:
            filename = f"./figures/{frame}.png"
            frame += 1
            filenames.append(filename)

            ax = plt.gca()
            ax.set_xlim3d(xmin=-7, xmax=7)
            ax.set_xlabel("X", fontsize=20)

            ax.set_ylim3d(ymin=-7, ymax=7)
            ax.set_ylabel("Y", fontsize=20)

            ax.set_zlim3d(zmin=0, zmax=14)
            ax.set_zlabel("Z", fontsize=20)

            angle = angle.reshape(-1, 1)
            self.plot(angle)

            ax.view_init(elev=elevation, azim=azimuth)
            ax.plot3D(x, y, z, 'gray')
            plt.savefig(filename)
            plt.cla()



        with imageio.get_writer(output_file, mode='I') as writer:
            for name in filenames:
                image = imageio.imread(name)
                writer.append_data(image)

        for name in filenames:
            remove(name)

    def animate1(self, angles: np.ndarray, output_file: str, azimuth: float,
                elevation: float,x,y,z) -> None:
        filenames = []
        frame = 0
        for angle in angles.T:
            filename = f"./figures/{frame}.png"
            frame += 1
            filenames.append(filename)

            ax = plt.gca()
            ax.set_xlim3d(xmin=-7, xmax=7)
            ax.set_xlabel("X", fontsize=20)

            ax.set_ylim3d(ymin=-7, ymax=7)
            ax.set_ylabel("Y", fontsize=20)

            ax.set_zlim3d(zmin=0, zmax=14)
            ax.set_zlabel("Z", fontsize=20)

            angle = angle.reshape(-1, 1)
            self.plot(angle)

            ax.view_init(elev=elevation, azim=azimuth)
            ax.plot3D(x, y, z, 'r')


    def perform_exercise(type: str) -> None:
        """
        Responsible for handling which exercise to run
        """
        pass

    def plot(
        self,
        theta: np.ndarray,
    ) -> np.ndarray:
        transformed_polygons = self.kinematic_map(theta)

        self.bodies['rigid'].plot('b')
        for type, polygon in transformed_polygons:
            if type == Plate.__name__:
                self.bodies['flexible'][0].plot_normal()
                polygon.plot('g')
            else:
                polygon.plot('r', "line")

    def kinematic_map(
            self, theta: np.ndarray) -> List[Tuple[str, Polygon, np.ndarray]]:
        return [part.kinematic_map(theta) for part in self.bodies['flexible']]

    def get_joint_angles(self, path: np.ndarray) -> np.ndarray:
        angles = np.array([[], [], []])

        for part in self.bodies['flexible']:
            if isinstance(part, Linkage):
                link, alpha, beta = part.get_joint_angles()
                angles = np.hstack((angles, np.vstack((link, alpha, beta))))

        return angles