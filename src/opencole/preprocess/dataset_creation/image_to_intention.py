from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Union

from datasets import ClassLabel
from pinjected import instance, injected, instances, Injected, Designed, providers
from tqdm import tqdm

from opencole.preprocess.dataset_creation.crello_instances import (
    crello_dataset_v4_0,
    crello_v4_0_train_sample,
)


@dataclass
class IntentionGenerationContext:
    title: str
    format: str
    keywords: list[str]
    texts: list[str]


@injected
def crello_sample_to_intention_generation_context(
    crello_format_features: ClassLabel, /, sample: dict
) -> IntentionGenerationContext:
    return IntentionGenerationContext(
        title=sample["title"],
        format=crello_format_features.int2str(sample["format"]),
        keywords=sample["keywords"],
        texts=[t for t in sample["text"] if t],
    )


@injected
async def a_cxt_to_intention(
    a_llm_for_intention, /, cxt: IntentionGenerationContext
) -> str:
    prompt = f"""
You are an excellent image analyst and capable of guessing an image designer’s intention. 
I will give you an image’s necessary information, including image title, image format, image keywords, all text contained in the image. 
Your task is to output a JSON formatted string.
This string contains a key value, intention, and the corresponding value is the user intention for designing this image. 
Please output the user intention from the perspective of the user using the image generation tool. 
Please note that the text information carried in the image may be helpful in the output results. 
Please include unique and necessary information in the entered text in the caption, such as website address, phone number, price, etc. 
Do not output any other irrelevant information. 
If needed, you can make reasonable guesses. 
Please refer to the example below for the desired format.
 
===== EXAMPLES =====
- Design an animated digital poster for a fantasy concert where Joe Hisaishi performs on a grand piano on a moonlit beach. The scene should be magical, with bioluminescent waves and a starry sky. The poster should invite viewers to an exclusive virtual event.
- Create a customer testimonial advertisement showcasing a pink clothing collection. The advertisement includes a positive review from Jenny Wilson.
- Create a poster advertising a corner spice shop with a wide range of spices and ingredients.
===== INPUT =====
title: {cxt.title}
format: {cxt.format}
keywords: {cxt.keywords}
texts: {cxt.texts}
===== ANSWER FORMAT=====
{{
    "intention": <your intention>
}}
"""
    return await a_llm_for_intention(prompt)


@instance
def crello_format_features(crello_dataset_v4_0):
    return crello_dataset_v4_0["train"].features["format"]


IdCxt = tuple[str, IntentionGenerationContext]


@injected
async def a_store_all_intentions(
    a_cxt_to_intention,
    a_map_progress,
    /,
    id_cxt_pairs: Union[Generator, list[IdCxt]],
    dst_root: Path,
):
    """
    Stores the detail json at dst_root / id[:3] / (id + ".json") for each image_id_pair
    :param image_id_pairs:
    :param dst_root:
    :return:
    """
    dst_root = Path(dst_root)
    dst_root.mkdir(exist_ok=True, parents=True)

    async def task(t):
        id, cxt = t
        intention = await a_cxt_to_intention(cxt)
        # logger.info(f"llm answer:{intention}")
        dst = dst_root / id[:3] / (id + ".json")
        dst.parent.mkdir(exist_ok=True, parents=True)
        dst.write_text(intention)

    async for item in a_map_progress(
        task,
        id_cxt_pairs,
        desc=f"storing intentions at {dst_root}",
    ):
        pass


@injected
async def a_crello_dataset_v4_0_all_id_cxt_pairs(
    crello_dataset_v4_0,
    crello_sample_to_intention_generation_context,
    /,
):
    for kind in tqdm(["train", "validation", "test"], desc="loading crello dataset"):
        d = crello_dataset_v4_0[kind]
        for item in tqdm(d, desc=f"loading crello samples from {kind}"):
            _id = item["id"]
            cxt = crello_sample_to_intention_generation_context(item)
            yield _id, cxt


check_train_class_label: Injected = crello_dataset_v4_0.proxy["train"].features[
    "format"
]
check_val_class_label: Injected = crello_dataset_v4_0.proxy["validation"].features[
    "format"
]

test_ig_cxt = crello_sample_to_intention_generation_context(crello_v4_0_train_sample)
crello_v4_0_pairs = a_crello_dataset_v4_0_all_id_cxt_pairs()
crello_v4_0_store_all_intentions = Designed.bind(
    a_store_all_intentions(
        # id_cxt_pairs=injected('pairs')
        a_crello_dataset_v4_0_all_id_cxt_pairs(),
        dst_root=injected("dst_root"),
    )
)

store_all_intentions = a_store_all_intentions(
    id_cxt_pairs=injected("pairs"), dst_root=injected("dst_root")
)


__meta_design__ = instances(overrides=providers())
