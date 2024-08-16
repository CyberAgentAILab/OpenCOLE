import argparse
import json
import logging
from pathlib import Path

from opencole.inference.tester.text2image import T2ITester
from opencole.schema import Detail, DetailV1

logger = logging.getLogger(__name__)


def main(DetailClass: type[Detail]):
    parser = argparse.ArgumentParser()
    T2ITester.register_args(parser)

    # weights
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--finetuned_unet_dir",
        type=str,
        help="Path to the fine-tuned UNet model for SDXL",
    )
    parser.add_argument(
        "--lora_weights_path", type=str, help="Path to the LoRA weights for SDXL"
    )
    # I/O
    parser.add_argument("--detail_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--first_n", type=int, default=None)
    args = parser.parse_args()
    logger.info(f"{args=}")

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    tester = T2ITester(
        **{
            k: v
            for (k, v) in vars(args).items()
            if k not in ["output_dir", "detail_dir", "first_n"]
        }
    )
    json_paths = sorted(Path(args.detail_dir).glob("*.json"))
    if args.first_n is not None:
        json_paths = json_paths[: args.first_n]
    if tester.use_chunking:
        json_paths = tester.get_chunk(json_paths)

    for json_path in json_paths:
        id_ = json_path.stem
        output_path = output_dir / f"{id_}.png"
        if output_path.exists():
            logger.info(f"Skipping {output_path=} since it already exists.")
            continue

        logger.info(f"Generating {output_path=}.")
        with open(json_path, "r") as f:
            detail = DetailClass(**json.load(f))

        output = tester(detail)
        output.save(str(output_path))


if __name__ == "__main__":
    main(DetailClass=DetailV1)
