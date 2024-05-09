"""
Organize a dataset of image-caption pairs for SimpleTuner.
"""

import argparse
import logging
import os
from functools import partial

import datasets as ds
from tqdm import tqdm
from transformers import AutoTokenizer

from layoutlib.hfds import hfds_factory, hfds_helper_factory
from layoutlib.hfds.util import filter_layer, is_expected_layer_type
from opencole.preprocess.helper import (
    adjust_text_length,
    check_image_size,
    is_layer_big_enough,
    save_image,
)
from opencole.renderer.renderer import ExampleRenderer
from opencole.schema import DetailV1

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

# see table 7 of COLE. 336 is For Llava1.5
# OUTPUT_RES_DICT = {"background": 256, "objects": 256, "non_text": 336}
TARGET_LAYER_NAMES = ["background", "objects", "non_text"]
NULL_TEXT = " "
TOKENIZER = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hfds_name", type=str, default="crello")
    parser.add_argument(
        "--input_hfds",
        type=str,
        default="cyberagent/opencole",
        help="a HuggingFace dataset containing intention and detail information",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        default="type",
        choices=["size", "type"],
    )
    parser.add_argument("--bg_size_thresh", type=float, default=0.8)
    parser.add_argument(
        "--target_layers", choices=["all"] + TARGET_LAYER_NAMES, default="non_text"
    )
    parser.add_argument("--max_size", type=int, default=None)
    # filter small images
    parser.add_argument(
        "--filter_type",
        type=str,
        default="area",
        choices=[
            "area",
        ],
    )
    parser.add_argument("--filter_thresh", type=int, default=0.5, help="in megapixels")
    args = parser.parse_args()

    if args.target_layers == "all":
        target_layers = ["background", "objects", "non_text"]
    else:
        target_layers = [args.target_layers]

    logger.info("Building mapping between id and detail...")
    id_detail_mapping: dict[str, DetailV1] = {}
    for split, dataset in ds.load_dataset(args.input_hfds).items():
        for example in dataset:
            id_detail_mapping[example["id"]] = DetailV1.from_hfds(example)
    logger.info("Done.")

    dataset_dict = hfds_factory(args.hfds_name)
    features = dataset_dict["train"].features
    type_feature = features["type"].feature
    hfds_helper = hfds_helper_factory(hfds_name=args.hfds_name, features=features)
    renderer = ExampleRenderer(features=hfds_helper.renderer_features)

    save_kwargs = {"flat_folder": True}
    for key in [
        "max_size",
    ]:
        save_kwargs[key] = getattr(args, key)

    # setup some utility functions for filtering
    non_text_layer_names = set(features["type"].feature.names)
    non_text_layer_names -= {hfds_helper.text_element_type_name}
    filter_is_non_text = partial(
        is_expected_layer_type,
        feature=type_feature,
        layer_names=non_text_layer_names,
    )
    filter_is_big_enough = partial(
        is_layer_big_enough, bg_size_thresh=args.bg_size_thresh
    )

    ids: dict[str, list[str]] = {k: [] for k in dataset_dict}
    for split in dataset_dict:
        dataset = dataset_dict[split]
        cnt = 0
        for i, example in enumerate(tqdm(dataset, desc=f"rendering {split}")):
            id_ = example["id"]

            if not check_image_size(
                example=example,
                hfds_helper=hfds_helper,
                filter_type=args.filter_type,
                filter_thresh=args.filter_thresh,
            ):
                continue

            detail: DetailV1 = id_detail_mapping[id_]
            ids[split].append(id_)
            layer_len = {}

            for layer in target_layers:
                filter_func = None
                if layer == "background":
                    if args.method == "size":
                        filter_func = filter_is_big_enough
                    elif args.method == "type":
                        filter_func = partial(
                            is_expected_layer_type,
                            feature=type_feature,
                            layer_names=["svgElement", "coloredBackground"],
                        )
                elif layer == "objects":
                    if args.method == "size":

                        def filter_func(x):  # type: ignore
                            return not filter_is_big_enough(x)  # type: ignore

                    elif args.method == "type":
                        filter_func = partial(
                            is_expected_layer_type,
                            feature=type_feature,
                            layer_names=["imageElement", "maskElement"],
                        )
                filter_kwargs = {
                    "example": example,
                }
                if filter_func:
                    filter_kwargs["filter_func"] = [filter_is_non_text, filter_func]
                else:
                    filter_kwargs["filter_func"] = [filter_is_non_text]
                example_filtered = filter_layer(**filter_kwargs)
                layer_len[layer] = example_filtered["length"]

                output_dir = f"{args.output_dir}/{layer}/{split}"
                save_image(
                    example=hfds_helper.convert_for_renderer(example_filtered),
                    renderer=renderer,
                    output_dir=output_dir,
                    **save_kwargs,
                )

                def _get(key: str) -> str:
                    return getattr(detail.captions, key)

                if layer == "non_text":
                    caption = detail.serialize_t2i_input()
                else:
                    caption = _get(layer)

                caption_original = caption
                caption = adjust_text_length(caption)

                with open(f"{output_dir}/{id_}.txt", "w") as f:
                    f.write(caption)
                with open(f"{output_dir}/{id_}_original.txt", "w") as f:
                    f.write(caption_original)

            if target_layers == ["background", "objects", "non_text"]:
                # check if no layer is missing
                assert (
                    layer_len["background"] + layer_len["objects"]
                    == layer_len["non_text"]
                )

            cnt += 1
            if os.environ.get("LOGLEVEL", "INFO") == "DEBUG" and cnt == 9:
                break

        logger.info(f"rendered {cnt} out of {len(dataset)} images for {split}")


if __name__ == "__main__":
    main()
