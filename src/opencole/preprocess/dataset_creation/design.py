from pathlib import Path

from pinjected import injected, instances, providers

from opencole.preprocess.dataset_creation.progress_util import (
    a_map_progress__tqdm_serial,
)
from opencole.preprocess.dataset_creation.util import device
from opencole.preprocess.dataset_creation.vision_llm_llava import a_vision_llm__llava

default_design = (
    instances(
        llava_model_label="llava-hf/llava-1.5-13b-hf",
        transformers_cache_path=Path("~/.cache").expanduser(),
    )
    + providers(
        device=device,
        a_vision_llm=a_vision_llm__llava,
        a_map_progress=a_map_progress__tqdm_serial,  # use tqdm for progress bar
        a_llm_for_json_fix=a_vision_llm__llava,  # change this to gpt4 or stg, in case the fix doesn't work
        opencole_cache_dir=injected("opencole_root_dir") / "cache",
        # opencole_n_batch_split=8,
        # opencole_batch_index=0,
    )
)
