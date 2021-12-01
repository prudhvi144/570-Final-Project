from math import sin, cos, pi
import numpy as np
from Polygon import Polygon


class Plate:
    def __init__(self, radius: float = 5) -> None:
        x_coords = [
            radius * angle for angle in
            [cos(pi / 6),
             cos(5 * pi / 6),
             cos(7 * pi / 6),
             cos(11 * pi / 6)]
        ]
        y_coords = [
            radius * angle for angle in
            [sin(pi / 6),
             sin(5 * pi / 6),
             sin(7 * pi / 6),
             sin(11 * pi / 6)]
        ]
        z_coords = [0, 0, 0, 0]
        self.vertices = np.array(list(zip(x_coords, y_coords, z_coords))).T
        self.polygon = Polygon(self.vertices)

    def plot(self, color: str = 'k') -> None:
        self.polygon.plot(color)