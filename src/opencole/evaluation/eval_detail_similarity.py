import argparse
import json
import logging
import os
from pathlib import Path

import datasets as ds
from tqdm import tqdm

from opencole.schema import DetailV1
from opencole.evaluation.sentence import compute_sentence_similarity_over_pairs


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_hfds", type=str, required=True)
    parser.add_argument(
        "--detail_dir",
        type=str,
        required=True,
        help="Stored as json following the schema of `opencole.schema.Detail`",
    )
    parser.add_argument(
        "--target_attribute",
        type=str,
        required=True,
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--model_name_or_path", type=str, default="all-MiniLM-L6-v2")

    args = parser.parse_args()
    logger.info(f"{args=}")

    dataset = ds.load_dataset(args.input_hfds, split=args.split)
    gt_examples = {}
    for example in tqdm(dataset, desc="Loading ground truth examples"):
        id_ = example["id"]
        detail = example
        gt_examples[id_] = detail

    pred_examples = {}
    for json_path in tqdm(
        list(Path(args.detail_dir).glob("*.json")), desc="Loading predictions"
    ):
        id_ = json_path.stem
        if id_ not in gt_examples:
            continue
        with open(json_path, "r") as f:
            example = DetailV1(**json.load(f)).to_hfds()
            example["id"] = id_
            pred_examples[id_] = example

    pairs = []
    for id_, pred_example in tqdm(pred_examples.items()):
        pairs.append([pred_example, gt_examples[id_]])
    similarity = compute_sentence_similarity_over_pairs(args.model_name_or_path, pairs)
    logger.info(f"Mean similarity: {similarity=}")


if __name__ == "__main__":
    main()
