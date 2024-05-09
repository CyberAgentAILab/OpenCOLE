# Note: this script should run in the same environment as the LLaVA model
import argparse
import logging
import os

import torch
import torch.nn as nn
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
)
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument("--model_base", type=str, default="liuhaotian/llava-v1.5-7b")
    args = parser.parse_args()
    model = load_model(model_path=args.model_path, model_base=args.model_base)
    torch.save(model.state_dict(), "model_state_dict.bin")


def load_model(model_path: str, model_base: str) -> nn.Module:
    """
    This script mostly replicates the load_pretrained_model function (for LLaVA LoRA only) from the LLaVA repository.
    https://github.com/haotian-liu/LLaVA/blob/main/llava/model/builder.py#L26
    """
    assert os.path.exists(model_path), f"{model_path} does not exist"
    assert "lora" in model_path.lower(), f"{model_path} does not contain 'lora'"
    assert "llava" in model_path.lower(), f"{model_path} does not contain 'llava'"

    device = torch.device("cuda")

    lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    print("Loading LLaVA from base model...")
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_base,
        low_cpu_mem_usage=True,
        config=lora_cfg_pretrained,
        **kwargs,
    )
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(
            torch.empty(
                token_num,
                tokem_dim,
                device=device,
                dtype=model.dtype,
            )
        )
        model.model.embed_tokens.weight = torch.nn.Parameter(
            torch.empty(
                token_num,
                tokem_dim,
                device=device,
                dtype=model.dtype,
            )
        )

    non_lora_trainables = torch.load(
        os.path.join(model_path, "non_lora_trainables.bin"),
        map_location="cpu",
    )
    non_lora_trainables = {
        (k[11:] if k.startswith("base_model.") else k): v
        for k, v in non_lora_trainables.items()
    }
    if any(k.startswith("model.model.") for k in non_lora_trainables):
        non_lora_trainables = {
            (k[6:] if k.startswith("model.") else k): v
            for k, v in non_lora_trainables.items()
        }
    model.load_state_dict(non_lora_trainables, strict=False)
    print(f"{model.dtype=}")

    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(
        model,
        model_path,
        device_map="auto",
    )
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)

    return model


if __name__ == "__main__":
    main()
