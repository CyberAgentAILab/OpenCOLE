import argparse
import logging
import os
from typing import TypeAlias

import torch
from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    CLIPImageProcessor,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)

StateDict: TypeAlias = dict[str, torch.Tensor]

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
}


def main() -> None:
    """
    Convert weights obtained from original LLaVA model to Hugging Face format.
    See https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/convert_llava_weights_to_hf.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="dumped state_dict from dump_full_llava_weight_dicts",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Location on local"
    )
    parser.add_argument(
        "--output_hub_path",
        type=str,
        default=None,
        help="Location on the hub of the converted model",
    )
    parser.add_argument(
        "--text_model_id",
        help="Hub location of the text model",
        default="lmsys/vicuna-7b-v1.5",
    )
    parser.add_argument(
        "--vision_model_id",
        help="Hub location of the vision model",
        default="openai/clip-vit-large-patch14-336",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token for write. (see https://huggingface.co/docs/hub/en/security-tokens)",
    )
    parser.add_argument("--is_public", action="store_true")
    args = parser.parse_args()
    logger.info(f"{args=}")

    if args.output_hub_path is not None:
        assert args.token is not None, "Please provide a token for pushing to the hub."
        assert (
            args.output_dir is None
        ), "Please specify either output_dir or output_hub_path."
    if args.output_dir is not None:
        assert (
            args.output_hub_path is None
        ), "Please specify either output_dir or output_hub_path."

    model, processor = _convert_llava_llama_to_hf(
        text_model_id=args.text_model_id,
        vision_model_id=args.vision_model_id,
        model_path=args.model_path,
    )

    if args.output_dir is not None:
        model.save_pretrained(
            args.output_dir,
            max_shard_size="5GB",
            safe_serialization=True,
        )
        processor.save_pretrained(args.output_dir)

    if args.output_hub_path is not None:
        model.push_to_hub(
            args.output_hub_path, private=not args.is_public, token=args.token
        )
        processor.push_to_hub(
            args.output_hub_path, private=not args.is_public, token=args.token
        )


def _convert_state_dict_to_hf(state_dict: StateDict) -> StateDict:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value
    return new_state_dict


def _convert_llava_llama_to_hf(
    text_model_id: str, vision_model_id: str, model_path: str
) -> tuple[LlavaForConditionalGeneration, LlavaProcessor]:
    """
    Note: currently designed to work with lla on top of liuhaotian/llava-v1.5-7b lmsys/vicuna-7b-v1.5
    """
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)
    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(
        AddedToken("<image>", special=True, normalized=False), special_tokens=True
    )
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    image_processor = CLIPImageProcessor.from_pretrained(vision_model_id)
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    config = LlavaConfig(text_config=text_config)
    config.pad_token_id = 32001

    with torch.device("meta"):
        model = LlavaForConditionalGeneration(config)

    # Pad to 64 for performance reasons
    pad_shape = 64

    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = _convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, strict=True, assign=True)

    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        mu, covariance_matrix=1e-5 * sigma
    )

    # We add an image token so we resize the model
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    model.language_model.model.embed_tokens.weight.data[32000:] = torch.stack(
        tuple(
            (
                dist.sample()
                for _ in range(
                    model.language_model.model.embed_tokens.weight.data[32000:].shape[0]
                )
            )
        ),
        dim=0,
    )
    model.language_model.lm_head.weight.data[32000:] = torch.stack(
        tuple(
            (
                dist.sample()
                for _ in range(
                    model.language_model.lm_head.weight.data[32000:].shape[0]
                )
            )
        ),
        dim=0,
    )
    return model, processor


if __name__ == "__main__":
    main()
