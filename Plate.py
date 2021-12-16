from typing import Tuple
from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt
import Polygon as polygon


class Plate:
    def __init__(self, radius: float, height: float) -> None:

        self.radius = radius
        self.height = height
        self.normal_vec = None
        if height > 0:
            self.normal_vec = np.cross([1, 0, 0], [0, 1, 0])

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
        w_T_tp = np.vstack((0, 0, self.height))

        transformed = np.array([[], [], []])
        for vertex in self.vertices.T:
            vertex = vertex.reshape(-1, 1)
            vertex_transf = w_R_tp @ vertex + w_T_tp
            transformed = np.hstack((transformed, vertex_transf))

        x_point: np.ndarray = w_R_tp @ np.vstack((1, 0, 0))  #+ w_T_tp
        y_point: np.ndarray = w_R_tp @ np.vstack((0, 1, 0))  #+ w_T_tp
        self.normal_vec = np.vstack((np.cross(x_point.ravel(),
                                              y_point.ravel())))

        return ("Plate", polygon.Polygon(transformed))

    def plot(self, color: str = 'k') -> None:
        self.polygon.plot(color)

    def plot_normal(self) -> None:
        ax = plt.gca()
        # vector starting from origin of top plate
        ax.quiver([0], [0], [self.height], [self.normal_vec[0, 0]],
                  [self.normal_vec[1, 0]], [self.normal_vec[2, 0]],
                  length=10,
                  color='g',
                  linewidth=2)
