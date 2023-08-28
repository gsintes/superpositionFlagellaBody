"""Object for the information of the mire."""

import json

class MireInfo:
    """
    Charectize the separation and displacement between the red and green image in the splitted beam configuration.
    
    middle_line (int): x-separation between the two colors.
    displacement (Tuple[int, int]): The vector of translation between the two shifted images. Green is the reference.
    """
    def __init__(self, *args) -> None:
        if len(args) == 2:
            self.middle_line = args[0]
            self.displacement = args[1]
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, dict):
                self.middle_line = arg["middle_line"]
                self.displacement = arg["displacement"]
            if isinstance(arg, str):
                with open(arg) as f:
                    data = json.load(f)
                    self.middle_line = data["middle_line"]
                    self.displacement = data["displacement"]

    def delta_x(self) -> int:
        """Return the displacement in x"""
        return self.displacement[0]
    
    def delta_y(self) -> int:
        """Return the displacement in y"""
        return self.displacement[1]
    
    def save(self, file: str) -> None:
        """Save the mire info in a json file."""
        with open(file, "w", encoding="utf-8") as outfile:
            outfile.write("")
            json.dump(self.__dict__, outfile, indent=4)

    def __repr__(self) -> str:
        return f"Mire info:\n middle_line: {self.middle_line}\n Displacement: {self.displacement}"