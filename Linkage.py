from operator import le
from typing import Tuple
from math import sin, cos, asin, sqrt, pi
import numpy as np
import Polygon as polygon


class Linkage:
    def __init__(self, radius: float, height: float,
                 link_angle: float) -> None:

        self.radius = radius
        self.height = height
        self.link_angle = link_angle
        self.alpha = pi / 4
        self.beta = pi / 4
        self.link_length = sqrt(self.height**2 / 2)

        cos_angle = radius * cos(self.link_angle)
        sin_angle = radius * sin(self.link_angle)

        x_coords = [
            cos_angle, cos_angle + (self.link_length * cos(self.alpha)),
            cos_angle
        ]
        y_coords = [
            sin_angle, sin_angle + (self.link_length * sin(self.alpha)),
            sin_angle
        ]
        z_coords = [0, self.height / 2, 0]

        self.vertices = np.array(list(zip(x_coords, y_coords, z_coords))).T
        self.polygon = polygon.Polygon(self.vertices)

    def kinematic_map(self, theta: np.ndarray) -> Tuple[str, polygon.Polygon]:
        x_axis_rot = theta[0, 0]
        y_axis_rot = theta[1, 0]

        w_R_tp = polygon.rot3d(x_axis_rot, y_axis_rot)
        end_effector_point = w_R_tp @ np.vstack(
            self.vertices[:, -1]) + np.vstack((0, 0, self.height))

        new_z = end_effector_point[2, 0] / 2
        self.alpha = asin(new_z / self.link_length)
        self.beta = 2 * self.alpha

        distance_to_motor = self.radius + self.link_length * cos(self.alpha)
        middle_point = np.vstack(
            (distance_to_motor * cos(self.link_angle),
             distance_to_motor * sin(self.link_angle), new_z))

        # bottom point doesn't change
        bottom_point = np.vstack(self.vertices[:, 0])

        return ("Linkage",
                polygon.Polygon(
                    np.hstack(
                        (bottom_point, middle_point, end_effector_point))))

    def plot(self, style: str = 'b') -> None:
        self.polygon.plot(style, "line")

    def get_joint_angles(self) -> Tuple[float, float, float]:
        return (self.link_angle, self.alpha, self.beta)
