import math
from copy import deepcopy
from typing import Any

import datasets as ds

from layoutlib.util import clamp

from .base import BaseHFDSHelper
from .util import Example

ADDITIONAL_FONT_PROPERTIES = {
    "bold": {"default": False, "dtype": "bool"},
    "italic": {"default": False, "dtype": "bool"},
}


class CrelloHFDSHelper(BaseHFDSHelper):
    name = "crello"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def normalize(self, example: Example) -> Example:
        canvas_width, canvas_height = self.get_canvas_size(example)
        output = {}

        # special
        if "angle" in example:
            angle_ = [v / (2.0 * math.pi) for v in example["angle"]]
            output["angle"] = [a - math.floor(a) for a in angle_]  # move to 0.0~1.0

        # already normalized, make sure it's in the range
        # (it may be out-of-bound due to the noise in the original data)
        # for attribute in ["left", "top", "width", "height"]:
        #     vmin, vmax = self.get_clamp_params(attribute)
        #     if attribute in example:
        #         output[attribute] = [
        #             clamp(v, vmin=vmin, vmax=vmax) for v in example[attribute]
        #         ]

        # consider canvas_width
        for attribute in [
            "letter_spacing",
        ]:
            if attribute in example:
                vmin, vmax = self.get_clamp_params(attribute)
                output[attribute] = [
                    clamp(v / canvas_width, vmin=vmin, vmax=vmax)
                    for v in example[attribute]
                ]

        # consider canvas_height
        for attribute in ["font_size"]:
            if attribute in example:
                vmin, vmax = self.get_clamp_params(attribute)
                output[attribute] = [
                    clamp(v / canvas_height, vmin=vmin, vmax=vmax)
                    for v in example[attribute]
                ]

        for attribute in example:
            if attribute not in output:
                feature = self.features[attribute]
                if (
                    isinstance(feature, ds.Sequence)
                    and isinstance(feature.feature, ds.Value)
                    and feature.feature.dtype.startswith("float")
                ):
                    vmin, vmax = self.get_clamp_params(attribute)
                    output[attribute] = [
                        clamp(v, vmin=vmin, vmax=vmax) for v in example[attribute]
                    ]
                else:
                    output[attribute] = example[attribute]

        return output

    def denormalize(
        self,
        example: Example,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
    ) -> Example:
        if canvas_width is None or canvas_height is None:
            canvas_width, canvas_height = self.get_canvas_size(example)
        output = {}

        # special
        for attribute in [
            "angle",
        ]:
            if attribute in example:
                output[attribute] = [v * 2.0 * math.pi for v in example[attribute]]

        # consider canvas_width
        for attribute in ["letter_spacing"]:
            if attribute in example:
                output[attribute] = [v * canvas_width for v in example[attribute]]

        # consider canvas_height
        for attribute in ["font_size"]:
            if attribute in example:
                output[attribute] = [v * canvas_height for v in example[attribute]]

        for attribute in example:
            if attribute not in output:
                output[attribute] = example[attribute]

        return output

    def get_canvas_size(self, example: Example) -> tuple[int, int]:
        canvas_width = int(
            self.features["canvas_width"].int2str(example["canvas_width"])
        )
        canvas_height = int(
            self.features["canvas_height"].int2str(example["canvas_height"])
        )
        return canvas_width, canvas_height

    @property
    def text_element_type_name(self) -> str:
        return "textElement"

    @property
    def renderer_features(self) -> ds.Features:
        """
        Our renderer assumes a bit different feature structure.
        This method returns the feature structure for the renderer.
        """
        if not hasattr(self, "_renderer_features"):
            renderer_features = deepcopy(self.features)
            renderer_features["canvas_width"] = ds.Value("int32")
            renderer_features["canvas_height"] = ds.Value("int32")

            names = renderer_features["type"].feature.names
            names[names.index("textElement")] = "TextElement"
            renderer_features["type"] = ds.Sequence(ds.ClassLabel(names=names))

            # text properties
            del renderer_features["color"]
            renderer_features["text_color"] = ds.Sequence(
                ds.Sequence(ds.Value("string"))
            )
            renderer_features["text_line"] = ds.Sequence(ds.Sequence(ds.Value("int32")))
            renderer_features["capitalize"] = ds.Sequence(ds.Value("string"))

            # font properties
            for key, prop in ADDITIONAL_FONT_PROPERTIES.items():
                renderer_features[f"font_{key}"] = ds.Sequence(
                    ds.Sequence(ds.Value(prop["dtype"]))
                )

            self._renderer_features = renderer_features
        return self._renderer_features

    def convert_for_renderer(
        self,
        example: dict,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
    ) -> Example:
        example_new = deepcopy(example)
        N = example_new["length"]

        if canvas_width is not None or canvas_height is not None:
            example_new["canvas_width"] = canvas_width
            example_new["canvas_height"] = canvas_height
        else:
            for key in ["canvas_height", "canvas_width"]:
                example_new[key] = int(self.features[key].int2str(example_new[key]))

        # set initial values
        example_new["text_color"] = [[] for _ in range(N)]
        example_new["text_line"] = [[] for _ in range(N)]
        for key in ADDITIONAL_FONT_PROPERTIES:
            example_new[f"font_{key}"] = [[] for _ in range(N)]

        for i in range(example_new["length"]):
            color = example_new["color"][i]
            if (
                self.features["type"].feature.int2str(example_new["type"][i])
                == "textElement"
            ):
                text = example_new["text"][i]

                text_line, line_index = [], 0
                for char in text:
                    if char == "\n":
                        line_index += 1
                    text_line.append(line_index)
                example_new["text_line"][i] = text_line

                example_new["text_color"][i] = [
                    f"rgba({int(color[0])}, {int(color[1])}, {int(color[2])}, 1)"
                    for _ in range(len(text))
                ]

                for key, prop in ADDITIONAL_FONT_PROPERTIES.items():
                    example_new[f"font_{key}"][i] = [
                        prop["default"] for _ in range(len(text))
                    ]

        example_new["capitalize"] = [
            self.features["capitalize"].feature.int2str(num).lower() == "true"
            for num in example_new["capitalize"]
        ]

        for key in ["width", "left"]:
            example_new[key] = [
                clamp(v) * example_new["canvas_width"] for v in example_new[key]
            ]
        for key in ["height", "top"]:
            example_new[key] = [
                clamp(v) * example_new["canvas_height"] for v in example_new[key]
            ]

        example_new["angle"] = [v * 180 / math.pi for v in example_new["angle"]]

        del example_new["color"]

        return example_new
