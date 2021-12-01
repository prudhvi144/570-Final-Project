from Plate import Plate
from Linkage import Linkage


class Neckbrace:
    def __init__(self) -> None:
        self.bodies = {
            "bottom": Plate(),
            "top": Plate(),
            "arm_30": Linkage(30),
            "arm_150": Linkage(150),
            "arm_210": Linkage(210),
            "arm_330:": Linkage(330)
        }

        self.potential = None

    def perform_exercise(type: str) -> None:
        """
        Responsible for handling which exercise to run
        """
        pass
