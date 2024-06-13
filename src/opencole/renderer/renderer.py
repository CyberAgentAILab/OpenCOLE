import logging
from typing import Any, Callable, Dict, Optional, Tuple

import datasets
import huggingface_hub
import skia

import opencole.renderer.image_utils as image_utils
import opencole.renderer.text_utils as text_utils
from opencole.renderer.fonts import FontManager
from opencole.renderer.schema import TextElement

logger = logging.getLogger(__name__)


class ExampleRenderer(object):
    def __init__(self, features: datasets.Features, fonts_path: Optional[str] = None):
        self.features = features
        self.font_manager = FontManager(fonts_path or self._fetch_fonts_from_hub())

    def _fetch_fonts_from_hub(self) -> str:
        """Load fonts."""
        return huggingface_hub.hf_hub_download(
            repo_id="cyberagent/opencole",
            filename="resources/fonts.pickle",
            repo_type="dataset",
        )

    def decode_class_label(self, example: dict) -> dict:
        """Apply `int2str` to all `datasets.ClassLabel` features in `example`."""

        def _get_decode_fn(feature: Any) -> Callable:
            if isinstance(feature, datasets.ClassLabel):
                return feature.int2str
            elif isinstance(feature, datasets.Sequence):
                return _get_decode_fn(feature.feature)
            return lambda x: x

        output = {}
        for key, feature in self.features.items():
            decode_fn = _get_decode_fn(feature)
            output[key] = decode_fn(example[key])
        return output

    def render(
        self,
        example: Dict[str, Any],
        max_size: int = 360,
        render_text: bool = True,
    ) -> bytes:
        """Render a preprocessed example and return as JPEG bytes."""
        example = self.decode_class_label(example)
        canvas_width = example["canvas_width"]
        canvas_height = example["canvas_height"]
        scale, size = get_scale_size(canvas_width, canvas_height, max_size)
        surface = skia.Surface(size[0], size[1])
        with surface as canvas:
            canvas.scale(scale[0], scale[1])
            canvas.clear(skia.ColorWHITE)
            for i in range(example["length"]):
                with skia.AutoCanvasRestore(canvas):
                    canvas.translate(example["left"][i], example["top"][i])
                    if example["angle"][i] != 0.0:
                        canvas.rotate(
                            example["angle"][i],
                            example["width"][i] / 2.0,
                            example["height"][i] / 2.0,
                        )
                    if (
                        example["type"][i] == "TextElement"
                        and self.font_manager
                        and render_text
                    ):
                        element = TextElement(
                            uuid="",
                            type="textElement",
                            width=float(example["width"][i]),
                            height=float(example["height"][i]),
                            text=str(example["text"][i]),
                            fontSize=float(example["font_size"][i]),
                            font=str(example["font"][i]),
                            lineHeight=float(example["line_height"][i]),
                            textAlign=str(example["text_align"][i]),  # type: ignore
                            capitalize=bool(example["capitalize"][i]),
                            letterSpacing=float(example["letter_spacing"][i]),
                            boldMap=text_utils.generate_map(example["font_bold"][i]),
                            italicMap=text_utils.generate_map(
                                example["font_italic"][i]
                            ),
                            colorMap=text_utils.generate_map(example["text_color"][i]),
                            lineMap=text_utils.generate_map(example["text_line"][i]),
                        )
                        text_utils.render_text(canvas, self.font_manager, element)
                    else:
                        image = image_utils.convert_pil_image_to_skia_image(
                            example["image"][i]
                        )
                        src = skia.Rect(image.width(), image.height())
                        dst = skia.Rect(example["width"][i], example["height"][i])
                        canvas.drawImageRect(image, src, dst)
        return image_utils.encode_surface(surface, "jpeg")


def get_scale_size(
    width: float, height: float, max_size: Optional[int] = None
) -> Tuple[Tuple[float, float], Tuple[int, int]]:
    """Get scale factor for the image."""
    s = 1.0 if max_size is None else min(1.0, max_size / max(width, height))
    sx, sy = s, s
    # Ensure the size is not zero.
    if round(width * sx) <= 0:
        sx = 1.0 / width
    if round(height * sy) <= 0:
        sy = 1.0 / height
    size = (round(sx * width), round(sy * height))
    assert (
        size[0] > 0 and size[1] > 0
    ), f"Failed to get scale: {size=}, {width=}, {height=}"
    return (sx, sy), size
