import argparse
import json
import logging
import os
from pathlib import Path

import datasets as ds
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from opencole.schema import DetailV1

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
    parser.add_argument("--split", default="test")
    parser.add_argument("--model_name_or_path", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--target_attribute",
        type=str,
        default="description",
        choices=[
            "keywords",
            "description",
            "captions_background",
            "captions_objects",
            "headings_heading",
            "headings_sub_heading",
        ],
    )
    args = parser.parse_args()
    logger.info(f"{args=}")

    model = SentenceTransformer(args.model_name_or_path)

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

    scores = []
    for id_, pred_example in tqdm(pred_examples.items(), desc="Computing similarity"):
        gt_example = gt_examples[id_]
        sim = _compute_similarity(
            gt_example[args.target_attribute],
            pred_example[args.target_attribute],
            model,
        )
        scores.append(sim)

    N = len(scores)
    logger.info(f"Mean similarity for {N} samples: {sum(scores) / N}")


def _compute_similarity(
    sentences1: str, sentences2: str, model: SentenceTransformer
) -> float:
    sentences = [sentences1, sentences2]
    emb = model.encode(sentences)
    cos_sim = util.cos_sim(emb[0], emb[1])
    return float(cos_sim)


if __name__ == "__main__":
    main()
