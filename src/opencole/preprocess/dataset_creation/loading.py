from dataclasses import dataclass
from pathlib import Path

import numpy as np
import PIL
from matplotlib import pyplot as plt
from PIL.Image import Image


@dataclass
class IdImagePath:
    id: str
    path: Path

    def load(self) -> Image:
        image = PIL.Image.open(self.path)
        ary = np.array(image)
        # fill the A=0 pixels with white
        ary[ary[:, :, 3] == 0] = 255
        image = PIL.Image.fromarray(ary)
        image = image.convert("RGB")
        return image

    async def aload(self):
        return self.load()

    def show_mpl(self):
        plt.imshow(self.load())
        plt.show()
