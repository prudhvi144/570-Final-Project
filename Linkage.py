from math import sin, cos, sqrt, pi
import numpy as np
from Polygon import Polygon


class Linkage:
    def __init__(self, radius: float, height: float,
                 link_angle: float) -> None:

        self.link_angle = link_angle
        self.alpha = pi / 4
        self.beta = pi / 4
        self.link_length = sqrt(height**2 / 2)

        cos_angle = radius * cos(self.link_angle)
        sin_angle = radius * sin(self.link_angle)

        x_coords = [
            cos_angle, cos_angle + (self.link_length * cos(self.link_angle)),
            cos_angle
        ]
        y_coords = [
            sin_angle, sin_angle + (self.link_length * sin(self.link_angle)),
            sin_angle
        ]
        z_coords = [0, self.link_length * cos(self.alpha), height]

        self.vertices = np.array(list(zip(x_coords, y_coords, z_coords))).T
        self.polygon = Polygon(self.vertices)

    def plot(self, style: str = 'b') -> None:
        self.polygon.plot(style, "line")