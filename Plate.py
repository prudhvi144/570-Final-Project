from math import sin, cos, pi
import numpy as np
from matplotlib.patches import Circle
from Polygon import Polygon


class Plate:
    def __init__(self, radius: float, height: float) -> None:

        self.radius = radius
        circle_coords = np.linspace(0, 2 * pi, 50)

        x_coords = [radius * cos(angle) for angle in circle_coords]
        y_coords = [radius * sin(angle) for angle in circle_coords]
        z_coords = [height] * len(circle_coords)

        self.vertices = np.array(list(zip(x_coords, y_coords, z_coords))).T
        self.polygon = Polygon(self.vertices)

    def plot(self, color: str = 'k') -> None:
        self.polygon.plot(color)