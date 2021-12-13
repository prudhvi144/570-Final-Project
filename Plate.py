from typing import Tuple
from math import sin, cos, pi
from mpl_toolkits.mplot3d.axis3d import XAxis
import numpy as np
from matplotlib.patches import Circle
import Polygon as polygon


class Plate:
    def __init__(self, radius: float, height: float) -> None:

        self.radius = radius
        self.height = height

        circle_coords = np.linspace(0, 2 * pi, 50)
        x_coords = [radius * cos(angle) for angle in circle_coords]
        y_coords = [radius * sin(angle) for angle in circle_coords]
        z_coords = np.zeros(len(circle_coords))

        self.vertices = np.array(list(zip(x_coords, y_coords, z_coords))).T
        self.polygon = polygon.Polygon(self.vertices)

    def kinematic_map(self, theta: np.ndarray) -> Tuple[str, polygon.Polygon]:
        x_axis_rot = theta[0, 0]
        y_axis_rot = theta[1, 0]

        w_R_tp = polygon.rot3d(x_axis_rot, y_axis_rot)

        transformed = np.array([[], [], []])
        for vertex in self.vertices.T:
            vertex = vertex.reshape(-1, 1)
            vertex_transf = w_R_tp @ vertex + np.vstack((0, 0, self.height))
            transformed = np.hstack((transformed, vertex_transf))

        return ("Plate", polygon.Polygon(transformed))

    def plot(self, color: str = 'k') -> None:
        self.polygon.plot(color)