import argparse
import logging
import os
from pathlib import Path

import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from opencole.evaluation.clip import extract_features

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Evaluate the similarity between two sets of images or texts (having similar stems) using CLIP model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_A", type=str, required=True)
    parser.add_argument("--dir_B", type=str, required=True)
    parser.add_argument("--ext_A", type=str, default="png")
    parser.add_argument("--ext_B", type=str, default="png")
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="image and text feature extractor",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    logger.info(f"{args=}")

    model = CLIPModel.from_pretrained(args.clip_model)
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    files_A = list(Path(args.dir_A).glob(f"*.{args.ext_A}"))
    if os.environ.get("LOGLEVEL", "INFO") == "DEBUG":
        files_A = files_A[:5]
    N = len(files_A)

    scores = []
    for i in range(0, N, args.batch_size):
        tmp_A, tmp_B = [], []
        for j in range(i, min(i + args.batch_size, N)):
            file_A = files_A[j]
            file_B = Path(args.dir_B) / f"{file_A.stem}.{args.ext_B}"
            tmp_A.append(_load(str(file_A)))
            tmp_B.append(_load(str(file_B)))

        features_A = extract_features(tmp_A, args.ext_A, model, processor)
        features_B = extract_features(tmp_B, args.ext_B, model, processor)

        for j in range(len(features_A)):
            sim = F.cosine_similarity(features_A[j : j + 1], features_B[j : j + 1])
            scores.append(sim.item())

    print(sum(scores) / len(scores))


def _load(name: str) -> Image.Image | str:
    if name.endswith("png"):
        return Image.open(name)
    elif name.endswith("txt"):
        with open(name, "r") as f:
            return f.read()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
