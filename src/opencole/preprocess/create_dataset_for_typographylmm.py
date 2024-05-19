import argparse
import io
import json
import logging
import os
import random
from functools import partial
from pathlib import Path

import datasets as ds
from langchain.output_parsers import PydanticOutputParser
from PIL import Image
from tqdm import tqdm

import layoutlib
from layoutlib.hfds import hfds_factory, hfds_helper_factory
from layoutlib.hfds.clustering import clustering_default_weight_path_factory
from layoutlib.hfds.util import (
    extract_class_label_mappings,
    filter_layer,
    is_expected_layer_type,
)
from layoutlib.manager import LayoutManager
from layoutlib.schema import get_layout_pydantic_model
from opencole.model.llava import padding_layout_transform
from opencole.preprocess.language import is_standard
from opencole.renderer.renderer import ExampleRenderer
from opencole.schema import DetailV1
from opencole.util import add_sub_directory, get_crello_image_path, set_seed, TYPOGRAPHYLMM_INSTRUCTION

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

KEYS = ["canvas_width", "canvas_height"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hfds_name", type=str, default="crello")
    parser.add_argument(
        "--skip_showing_choices", type=str, nargs="*", default=["font"]
    )
    parser.add_argument(
        "--input_hfds",
        type=str,
        default="cyberagent/opencole",
        help="a HuggingFace dataset containing intention and detail information",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_num_text_layers", type=int, default=20)
    parser.add_argument(
        "--skip_save_image",
        action="store_true",
        help="skip saving images (useful when trying various annotation encodings)",
    )
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument(
        "--max_size",
        type=int,
        default=336,
        help="currently rendered images will be processed in Llava1.5",
    )
    parser.add_argument(
        "--element_order",
        type=str,
        default="sort_lexicographic",
        choices=["shuffle", "sort_lexicographic"],
    )
    # tokenizer settings
    parser.add_argument("--schema_name", type=str, default="typography_crello_default")
    parser.add_argument(
        "--weight_path",
        type=str,
        default=None,
        help="required if a tokenizer employes clustering",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    logger.info("Building mapping between id and detail...")
    id_detail_mapping: dict[str, DetailV1] = {}
    for split, dataset in ds.load_dataset(args.input_hfds).items():
        for example in dataset:
            id_detail_mapping[example["id"]] = DetailV1.from_hfds(example)
    logger.info("Done.")

    element_order_transform = getattr(
        layoutlib.hfds.util, f"{args.element_order}_transform"
    )

    output_dir = Path(args.output_dir)
    directories = {k: add_sub_directory(output_dir, k) for k in ["images", "json"]}

    dataset_dict = hfds_factory(args.hfds_name)
    features = dataset_dict["train"].features
    hfds_helper = hfds_helper_factory(hfds_name=args.hfds_name, features=features)
    type_feature = features["type"].feature
    text_layer_names = {hfds_helper.text_element_type_name}
    non_text_layer_names = set(features["type"].feature.names) - text_layer_names
    filter_is_text = partial(
        is_expected_layer_type,
        feature=type_feature,
        layer_names=text_layer_names,
    )
    filter_is_non_text = partial(
        is_expected_layer_type,
        feature=type_feature,
        layer_names=non_text_layer_names,
    )

    renderer = ExampleRenderer(features=hfds_helper.renderer_features)

    class_label_mappings = extract_class_label_mappings(features)
    manager = LayoutManager(
        schema_name=args.schema_name,
        weight_path=clustering_default_weight_path_factory(args.hfds_name),
        class_label_mappings=class_label_mappings,
    )
    layout_parser = PydanticOutputParser(
        pydantic_object=get_layout_pydantic_model(
            schema=manager.schema,
            class_label_mappings=manager.class_label_mappings,
            skip_showing_choices=args.skip_showing_choices,
        )
    )  # type: ignore
    logger.info(f"{manager.schema=}")
    logger.info(f"{manager.class_label_mappings=}")
    logger.info(f"{layout_parser.get_format_instructions()}")

    instruction = f"{TYPOGRAPHYLMM_INSTRUCTION} {layout_parser.get_format_instructions()}"

    for split in dataset_dict:
        annotations = []
        dataset = dataset_dict[split]
        for i, example in enumerate(
            tqdm(dataset, desc=f"tokenizing layouts in {split}")
        ):
            id_ = example["id"]

            if id_ not in id_detail_mapping:
                continue  # unfortunately, we could not label some of crello v4 dataset.

            W, H = hfds_helper.get_canvas_size(example)

            # skip if there is non-standard strings in an example
            if not all([is_standard(t) for t in example["text"]]):
                continue

            example_non_text = filter_layer(
                filter_func=filter_is_non_text, example=example
            )
            N = example_non_text["length"]
            if N <= 0 or N > args.max_num_text_layers:
                continue

            if not args.skip_save_image:
                image_bytes = renderer.render(
                    example=hfds_helper.convert_for_renderer(example_non_text),
                    max_size=args.max_size,
                )
                image = Image.open(io.BytesIO(image_bytes))
                image_path = get_crello_image_path(root=directories["images"], id_=id_)
                if not image_path.exists():
                    image.save(str(image_path))

            example_text = filter_layer(filter_func=filter_is_text, example=example)
            example_text = element_order_transform(example_text)
            example_text = padding_layout_transform(
                example_text, canvas_width=W, canvas_height=H
            )

            # replace "heading" part in detail to a set of texts
            # this is to force the model to accurately copy the input texts
            example_text = hfds_helper.normalize(example_text)
            layout_text = manager.hfds_to_layout_instance(example_text)
            texts = [e.text for e in layout_text.elements]  # type: ignore
            input_ = json.dumps(random.sample(texts, len(texts)))

            detail: DetailV1 = id_detail_mapping[id_]
            context = detail.serialize_tlmm_context()

            annotation = {
                "id": id_,
                "image": f"images/{id_[:3]}/{id_}.png",
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{instruction} Context: {context} Input: {input_}",
                    },
                    {"from": "gpt", "value": layout_text.json()},
                ],
            }

            annotations.append(annotation)

            if os.environ.get("LOGLEVEL", "INFO") == "DEBUG" and i == 9:
                break

        with open(str(directories["json"] / f"{split}.json"), "w") as f:
            json.dump(annotations, f, indent=4)

        # save a list of valid ids
        ids = [annotation["id"] for annotation in annotations]
        with open(str(directories["json"] / f"{split}_ids.txt"), "w") as f:
            logger.info(f"Valid ids in {split}: {len(ids)} out of {len(dataset)}")
            f.write("\n".join(ids))

    with open(str(directories["json"] / "preprocess_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    main()
