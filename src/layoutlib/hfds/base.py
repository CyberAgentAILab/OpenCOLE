import math

import datasets as ds

from .util import Example


class BaseHFDSHelper:
    """
    Handling dataset-specific operations in HFDS dataset.
    """

    name = "base"

    def __init__(self, features: ds.Features) -> None:
        self._features = features

    def __call__(self) -> dict:
        raise NotImplementedError

    def normalize(self, example: Example) -> Example:
        """
        Normalize each attribute in the example so that float values are between 0.0 and 1.0.
        """
        raise NotImplementedError

    def denormalize(
        self,
        example: Example,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
    ) -> Example:
        """
        Canvas width and height can be specified exteranally.
        """
        raise NotImplementedError

    def get_canvas_size(self, example: Example) -> tuple[int, int]:
        raise NotImplementedError

    def get_clamp_params(self, attribute: str) -> tuple[float, float]:
        """
        Set data-specific clamp parameters for normalization.
        """
        # TODO (nits): dataset-specific settings in the future?
        if attribute == "line_height":
            return 0.5, 5.0
        elif attribute == "angle":
            # note: simple clip may harm the visual quality.
            return -math.inf, math.inf
        elif attribute == "letter_spacing":
            return -0.005, 0.025
        else:
            return 0.0, 1.0

    @property
    def features(self) -> ds.Features:
        return self._features

    @property
    def text_element_type_name(self) -> str:
        raise NotImplementedError

    @property
    def renderer_features(self) -> ds.Features:
        return self.features

    def convert_for_renderer(
        self,
        example_in: dict,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
    ) -> Example:
        return example_in
