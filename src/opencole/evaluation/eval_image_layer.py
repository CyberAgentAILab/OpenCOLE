import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from opencole.evaluation.clip import DEVICE, extract_features

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
        default="google/siglip-so400m-patch14-384",
        help="image and text feature extractor",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--first_n", type=int, default=None)
    args = parser.parse_args()
    logger.info(f"{args=}")

    device_cpu = torch.device("cpu")

    model = AutoModel.from_pretrained(args.clip_model).to(DEVICE)
    processor = AutoProcessor.from_pretrained(args.clip_model)

    pairs = _get_url_pairs(args.dir_A, args.dir_B, args.ext_A, args.ext_B)
    if args.first_n is not None:
        pairs = pairs[: args.first_n]

    scores = []
    for i in range(0, len(pairs), args.batch_size):
        slice_ = slice(i, min(i + args.batch_size, len(pairs)))
        features_A = extract_features(
            [_load(str(p[0])) for p in pairs[slice_]], args.ext_A, model, processor
        ).to(device_cpu)
        features_B = extract_features(
            [_load(str(p[1])) for p in pairs[slice_]], args.ext_B, model, processor
        ).to(device_cpu)

        for j in range(len(features_A)):
            sim = F.cosine_similarity(features_A[j : j + 1], features_B[j : j + 1])
            scores.append(sim.item())

    print(sum(scores) / len(scores))


def _get_url_pairs(
    dir_A: str, dir_B: str, ext_A: str, ext_B: str
) -> list[tuple[str, str]]:
    """
    Get pairs of urls from two directories.
    """
    files_A = list(Path(dir_A).glob(f"*.{ext_A}"))
    files_B = set(list(Path(dir_B).glob(f"*.{ext_B}")))
    pairs = []
    for file_A in files_A:
        file_B = Path(dir_B) / f"{file_A.stem}.{ext_B}"
        if file_B in files_B:
            pairs.append((str(file_A), str(file_B)))
    return pairs


def _load(name: str) -> Image.Image | str:
    if name.endswith("png"):
        return Image.open(name).convert("RGB")
    elif name.endswith("txt"):
        with open(name, "r") as f:
            return f.read()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
