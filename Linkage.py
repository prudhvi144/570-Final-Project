from math import sin, cos, pi
import numpy as np
from Polygon import Polygon


class Linkage:
    def __init__(self, radius: float, angle: float) -> None:
        cos_angle = radius * cos(angle)
        sin_angle = radius * sin(angle)
        if cos_angle < 0:
            distance = -4.5
        else:
            distance = 4.5

        x_coords = [
            cos_angle, cos_angle + (distance * cos(pi / 4)) * sin(pi / 4),
            cos_angle
        ]
        y_coords = [sin_angle, sin_angle, sin_angle]
        z_coords = [0, 4.5 * cos(pi / 4) * cos(pi / 4), 4.5]

        self.vertices = np.array(list(zip(x_coords, y_coords, z_coords))).T
        self.polygon = Polygon(self.vertices)

    def plot(self, style):
        self.polygon.plot(style, "line")