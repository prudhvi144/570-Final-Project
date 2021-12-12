from math import pi
from typing import List
import numpy as np
from Plate import Plate
from Linkage import Linkage


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
                link_angles=List[float]):
        assert plate_radius > 0.0, 'Neckbrace radius must be greater than 0.0'
        assert height > 0.0, 'Neckbrace height must be greater than 0.0'
        assert len(link_angles) > 0, 'Number of links must be greater than 0'

    def perform_exercise(type: str) -> None:
        """
        Responsible for handling which exercise to run
        """
        pass

    def plot(self):
        i = 0
        colors = ['b', 'g']
        for name, parts in self.bodies.items():
            if name == "rigid":
                parts.plot(colors[i])
                i += 1

            else:
                for rigid_body in parts:
                    if isinstance(rigid_body, Plate):
                        rigid_body.plot(colors[i])
                        i += 1

                    if isinstance(rigid_body, Linkage):
                        rigid_body.plot('r')