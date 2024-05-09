"""
Organize a dataset of image-caption pairs for SimpleTuner.
"""

import io
import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Optional

from PIL import Image
from transformers import AutoTokenizer

from layoutlib.hfds import BaseHFDSHelper
from opencole.renderer.renderer import ExampleRenderer

logger = logging.getLogger(__name__)
TOKENIZER = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def is_layer_big_enough(example: dict, index: int, bg_size_thresh: float = 0.8) -> bool:
    flag = example["width"][index] >= bg_size_thresh
    flag |= example["height"][index] >= bg_size_thresh
    return bool(flag)


def check_image_size(
    example: dict, hfds_helper: BaseHFDSHelper, filter_type: str, filter_thresh: float
) -> bool:
    """
    Check if the image is big enough, which is essential for recent text-to-image models that can generate HD images.
    """
    W, H = hfds_helper.get_canvas_size(example)

    if filter_type == "area":
        thresh = filter_thresh * 1e6  # convert to pixels
        return bool((W * H) >= thresh)
    elif filter_type == "pixel":
        return bool(W >= filter_thresh and H >= filter_thresh)
    else:
        raise NotImplementedError


def adjust_text_length(caption: str, tokenizer: AutoTokenizer = TOKENIZER) -> str:
    """
    Typical text-to-image models can handle up to 77 tokens.
    This functions resamples the sentences in the given caption to fit the length.
    """
    data = deepcopy(caption)
    finished = False
    while not finished:
        current_length = len(tokenizer(data)["input_ids"])  # type: ignore
        if current_length > 77:
            # drop a random sentence
            sentences = data.split(". ")
            N = len(sentences)
            ind = random.randint(0, N - 1)
            data = ". ".join([sentences[i] for i in range(N) if i != ind])
        else:
            finished = True
    return data


def save_image(
    example: dict,
    renderer: ExampleRenderer,
    output_dir: str,
    flat_folder: bool = False,
    max_size: Optional[int] = None,
) -> None:
    id_ = example["id"]
    if flat_folder:
        # employ flat directory structure
        output_dir_ = Path(output_dir)
    else:
        # split into smaller chunks
        output_dir_ = Path(output_dir) / id_[:3]
    if not output_dir_.exists():
        output_dir_.mkdir(parents=True, exist_ok=True)
    image_path = output_dir_ / f"{id_}.png"
    if image_path.exists():
        return

    if max_size is None:  # set temporal canvas size
        max_size = 1024

    image_bytes = renderer.render(
        example=example,
        max_size=max_size,
    )
    image = Image.open(io.BytesIO(image_bytes))
    image.save(str(image_path))
