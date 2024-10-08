import argparse
import io
import json
import logging
import os
from pathlib import Path

import datasets as ds
from PIL import Image

from layoutlib.hfds import hfds_helper_factory, sample_example
from layoutlib.hfds.util import Example, fill_missing_values, get_default_value
from opencole.renderer.renderer import ExampleRenderer

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


DEFAULT_IMAGE_SIZE = (256, 256)


def main(tester: "BaseRendererTester") -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="directory containing a single image generated by text-to-image given a detail.",
    )
    parser.add_argument(
        "--typography_dir",
        type=str,
        required=True,
        help="directory containing a typography details in a JSON format. For attributes, please refer to cyberagent/crello dataset.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--first_n", type=int, default=None)
    args = parser.parse_args()

    typography_list = list(Path(args.typography_dir).glob("*.json"))
    image_list = list(Path(args.image_dir).glob("*.png"))
    assert len(typography_list) == len(image_list) and len(typography_list) > 0
    assert set([x.stem for x in typography_list]) == set([x.stem for x in image_list])
    typography_image_pairs = zip(sorted(typography_list), sorted(image_list))
    if args.first_n is not None:
        typography_image_pairs = list(typography_image_pairs)[: args.first_n]

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    for typography_path, image_path in typography_image_pairs:
        stem = typography_path.stem
        image = Image.open(str(image_path))
        with typography_path.open("r") as fp:
            typography = json.load(fp)

        try:
            output_image = tester(image, typography)
        except Exception as e:
            logger.error(f"Error in {stem=}: {e=}")
            output_image = image

        output_path = output_dir / f"{stem}.png"
        output_image.save(str(output_path))


class BaseRendererTester:
    """Basic interface for tester."""

    def __init__(self, hfds_name: str) -> None:
        _, self.features = sample_example(hfds_name)
        self.hfds_helper = hfds_helper_factory(
            hfds_name=hfds_name, features=self.features
        )

    def __call__(self, background: Image.Image, typography: Example) -> Image.Image:
        raise NotImplementedError


class RendererTester(BaseRendererTester):
    def __init__(self, hfds_name: str = "crello") -> None:
        super().__init__(hfds_name=hfds_name)
        self.renderer = ExampleRenderer(features=self.hfds_helper.renderer_features)

    def __call__(self, background: Image.Image, typography: Example) -> Image.Image:
        typography = self.insert_image_layer(typography, background)
        W, H = background.size
        typography = self.hfds_helper.convert_for_renderer(
            example=typography, canvas_width=W, canvas_height=H
        )
        image_bytes = self.renderer.render(typography, max_size=max(W, H))
        return Image.open(io.BytesIO(image_bytes))

    def insert_image_layer(self, example: Example, background: Image.Image) -> Example:
        # insert type
        N = len(example["left"])
        text_id = self.features["type"].feature.str2int(
            self.hfds_helper.text_element_type_name
        )
        image_id = self.features["type"].feature.str2int("imageElement")
        example["type"] = [text_id] * N
        example["image"] = [Image.new("RGBA", DEFAULT_IMAGE_SIZE)] * N

        # insert image
        example["image"] = [background.convert("RGBA")] + example["image"]
        example["type"] = [image_id] + example["type"]
        for key in ["left", "top", "angle"]:
            example[key] = [0.0] + example[key]
        for key in ["width", "height"]:
            example[key] = [1.0] + example[key]

        example["opacity"] = [1.0] * (N + 1)
        example["length"] = N + 1

        for key, feature in self.features.items():
            if isinstance(feature, ds.Sequence) and key in example:
                if len(example[key]) == N + 1:
                    # already filled by operations above
                    pass
                elif len(example[key]) == N:
                    example[key] = [get_default_value(feature.feature)] + example[key]

        example = fill_missing_values(example, self.features)
        return example


if __name__ == "__main__":
    main(tester=RendererTester())
