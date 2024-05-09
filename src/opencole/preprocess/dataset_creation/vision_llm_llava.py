import os
from pathlib import Path

import pandas as pd
import PIL
import torch
from loguru import logger
from pinjected import injected, instance, instances
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor


@instance
def setup_transformer_cache_path_result(transformers_cache_path: Path, logger):
    os.environ["TRANSFORMERS_CACHE"] = str(
        transformers_cache_path.expanduser().absolute()
    )
    from transformers import file_utils

    # idk why but this doesn't seem to have effect!
    logger.info(f"using transformers cache path:{file_utils.default_cache_path}")
    logger.info(f"current environment variable:{os.environ.get('TRANSFORMERS_CACHE')}")
    # list files in cache dir:
    for f in os.listdir(file_utils.default_cache_path):
        logger.info(f"file in cache dir:{f}")
    # list files in env dir:
    for f in os.listdir(str(transformers_cache_path.expanduser().absolute())):
        logger.info(f"file in env dir:{f}")
    file_utils.default_cache_path = str(transformers_cache_path.expanduser().absolute())
    return transformers_cache_path


@instance
def llava_model(transformers_cache_path: Path, llava_model_label: str):
    model = LlavaForConditionalGeneration.from_pretrained(
        llava_model_label, device_map="auto", cache_dir=transformers_cache_path
    )
    return model


@instance
def llava_processor(transformers_cache_path: Path, llava_model_label) -> LlavaProcessor:
    processor = AutoProcessor.from_pretrained(
        llava_model_label, cache_dir=transformers_cache_path
    )
    return processor


@injected
async def a_vision_llm__llava(
    llava_model, llava_processor, device, /, text: str, images: list[PIL.Image] = None
):
    start = pd.Timestamp.now()
    inputs = llava_processor(text=text, images=images, return_tensors="pt")
    fixed_inputs = dict()
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            fixed_inputs[k] = v.to(device)
        else:
            fixed_inputs[k] = v

    generate_ids = llava_model.generate(**fixed_inputs, max_length=1000)
    res = llava_processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    end = pd.Timestamp.now()
    src_len = len(text.replace("<image>", ""))
    generated_text = res[src_len:]
    token_per_sec = len(generate_ids) / (end - start).total_seconds()
    logger.info(
        f"llava took {end - start} for {len(generate_ids)} tokens -> {token_per_sec} token/sec"
    )
    return generated_text


__meta_design__ = instances()
