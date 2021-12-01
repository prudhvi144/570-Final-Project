from math import pi
from Plate import Plate
from Linkage import Linkage


class Neckbrace:
    def __init__(self, plate_radius: float) -> None:
        self.plate_radius = plate_radius

        self.bodies = {
            "base": Plate(self.plate_radius),
            "top": Plate(self.plate_radius),
            "arm_30": Linkage(self.plate_radius, pi / 6),
            "arm_150": Linkage(self.plate_radius, 5 * pi / 6),
            "arm_210": Linkage(self.plate_radius, 7 * pi / 6),
            "arm_330:": Linkage(self.plate_radius, 11 * pi / 6)
        }
        # TODO: Fix kinematic aspect
        self.bodies['top'].vertices[2, :] = 4 * [4.5]

        self.potential = None

    def perform_exercise(type: str) -> None:
        """
        Responsible for handling which exercise to run
        """
        pass

    def plot(self):
        i = 0
        colors = ['b', 'g']
        for name, rigid_body in self.bodies.items():
            if isinstance(rigid_body, Plate):
                rigid_body.plot(colors[i])
                i += 1

            if isinstance(rigid_body, Linkage):
                rigid_body.plot('r')