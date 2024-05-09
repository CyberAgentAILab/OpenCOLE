import argparse
import io
import logging
import os

import datasets as ds
import huggingface_hub
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def _get_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Hugging Face API token for write. (see https://huggingface.co/docs/hub/en/security-tokens)",
    )
    parser.add_argument("--is_public", action="store_true")
    parser.add_argument("--input_dir", type=str, required=True)
    return parser


def push_finetuned_sdxl() -> None:
    """
    Pusing the only UNet model to the hub
    """
    parser = _get_base_parser()
    args = parser.parse_args()
    torch_dtype = torch.bfloat16

    unet = UNet2DConditionModel.from_pretrained(args.input_dir, torch_dtype=torch_dtype)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch_dtype
    )
    pipeline.push_to_hub(
        repo_id=args.repo_id, private=not args.is_public, token=args.token
    )


def push_hfds() -> None:
    # TODO: set detailed metadata for public release
    parser = _get_base_parser()
    parser.add_argument("--fonts_path", type=str, required=True)
    args = parser.parse_args()

    _DESCRIPTION = """
    The dataset contains additional synthetically generated annotations for each sample in crello v4.0.0 dataset (https://huggingface.co/datasets/cyberagent/crello).
    """

    _CITATION = """
    @inproceedings{inoue2024opencole,
      title={{OpenCOLE: Towards Reproducible Automatic Graphic Design Generation}},
      author={Naoto Inoue and Kento Masui and Wataru Shimoda and Kota Yamaguchi},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
      year={2024},
    }
    """

    _HOMEPAGE = f"https://huggingface.co/datasets/{args.repo_id}"

    _VERSION = "1.0.0"

    dataset = ds.load_from_disk(args.input_dir)

    # Metadata setup, though seems not working. A bug?
    for split in dataset.keys():
        dataset[split].info.description = _DESCRIPTION
        dataset[split].info.citation = _CITATION
        dataset[split].info.homepage = _HOMEPAGE
        dataset[split].info.version = _VERSION

    dataset.push_to_hub(
        repo_id=args.repo_id, private=not args.is_public, token=args.token
    )

    with open(args.fonts_path, "rb") as fp:
        logger.info("Uploading fonts.pickle...")
        huggingface_hub.upload_file(
            path_or_fileobj=io.BufferedReader(fp),  # type: ignore
            path_in_repo="resources/fonts.pickle",
            repo_id=args.repo_id,
            repo_type="dataset",
            token=args.token,
        )

    # Finally, create a tag.
    huggingface_hub.create_tag(
        repo_id=args.repo_id,
        repo_type="dataset",
        tag=_VERSION,
        token=args.token,
    )


if __name__ == "__main__":
    push_finetuned_sdxl()
