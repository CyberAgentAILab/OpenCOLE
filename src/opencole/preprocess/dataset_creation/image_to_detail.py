# hmm, why is llava not importable?
import asyncio
import json
import traceback
from json import JSONDecodeError
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, List, Union

import PIL
from loguru import logger
from openai import BadRequestError
from PIL.Image import Image
from pinjected import Injected, injected, instances
from returns.result import Failure
from tqdm import tqdm

from opencole.preprocess.dataset_creation.loading import IdImagePath
from opencole.schema import Captions, DetailV1, Headings

IMAGE_EXTENTIONS = [".jpg", ".jpeg", ".png"]

VisionLLM = Callable[[str, list[Image]], Awaitable[str]]
VisionLLMJson = Callable[[str, list[Image]], Awaitable[dict]]
GLOBAL_CAPTION_PROMPT = """
<image>
Write a very long informative global caption of the provided image. 
The caption should include what the image is about, what are the objects in the image, and what is the context of the image.
Do not include explanation about the small text which cannot be read clearly.
"""
KEYWORD_PROMPT = """
<image>
Please extract the keywords which best describes the provided image in comma separated format.
=== Example ===
green, plant, houseplant, decoration, pot, natural, decor, nature, home, room, flower, gardening, floral, flowerpot,
 houseplants, botanical, exotic, leaves, succulent, decorative, cactus, fresh, illustration, 
 armchair, pink, arrows, services, cute, creative, creativity, dizzy, post, instagram, ig, insta
===============
Now, extract the keywords that describes the provided image, each keyword should be one word.
"""

IMAGE_PROMPT = """
<image>
Please separate the background image and content objects in the provided image, 
and write a detailed explanation for each of them.
The answer must be in the following json format.
Do not include ``` or ```json in your answer.
=== Example ===
{
    "background": "<very detailed explanation of the background>",
    "objects": "<very detailed explanation of the objects>"
}
===============
"""
HEADINGS_PROMPT = """
<image>
Please extract the texts in the provided image, while separating them into heading and sub-heading.
The answer must be in the following json format.
Do not include ``` or ```json in your answer.
=== Example ===
{
    "heading": ["for plants", "Flora Store", "WITH ROOM"],
    "sub-heading": ["We can help you"]
}
===============
"""


@injected
async def a_fix_malformed_json(a_llm_for_json_fix, /, malformed_json_str, context=None):
    logger.info(
        f"fixing malformed json with llm...{malformed_json_str[:100]}(truncated)"
    )
    prompt = f"""
Please fix the syntax error in the following json. 
Only the corrected json should be answered without ```.
No comments are required, write a pure json text.
If multiple jsons are found, merge it.
Pay extra attention on double quotes for escaping.
===== CONTEXT =====
{context}
===== START =====
{malformed_json_str}
===== END =====
"""
    res = await a_llm_for_json_fix(prompt)
    logger.info(f"fixed json:{res}")
    return res


malformed_llm_answer = """
===============

{
    "background": "The background is a green and white checkered pattern.",
    "objects": "The objects are a green and white checkered pattern."
}
===============

{
    "background": "The background is a green and white checkered pattern.",
    "objects": "The objects are a green and white checkered pattern."
}
===============

{
    "background": "The background is a green and white checkered pattern.",
    "objects": "The objects are a green and white checkered pattern."
}
===============

{
    "background": "The background is a green and white checkered pattern.",
    "objects": "The objects are a green and white checkered pattern."
}
===============

{
    "background": "The background is a green and white checkered pattern.",
    "objects": "The objects are a green and white checkered pattern."
}
===============

{
    "background": "The background is a green and white checkered pattern.",
    "objects": "The objects are a green and white checkered pattern."
}
===============

{
"""


@injected
def extract_json_from_llm_answer(text):
    text = text.strip()
    text = text[text.find("{") : text.rfind("}") + 1]
    return json.loads(text)


@injected
async def a_fix_and_parse_llm_json(
    extract_json_from_llm_answer,
    a_fix_malformed_json,
    /,
    llm_answer: str,
    fix_context: str = None,
):
    res = llm_answer
    try:
        return extract_json_from_llm_answer(res)
    except Exception as e:
        logger.warning(f"failed to parse ==>\n{res}\n<==")
        refined = await a_fix_malformed_json(res, fix_context)
        try:
            return extract_json_from_llm_answer(refined)
        except Exception as e2:
            logger.error(f"failed to parse refined json again ==>\n{refined}\n<==")
            raise e2 from e


@injected
async def a_vision_llm_json(
    a_vision_llm: VisionLLM,
    a_fix_and_parse_llm_json,
    /,
    prompt,
    images,
    fix_context=None,
) -> dict:
    res = await a_vision_llm(prompt, images)
    return await a_fix_and_parse_llm_json(res, fix_context=fix_context)


@injected
async def a_image_to_global_caption(a_vision_llm: VisionLLM, /, image) -> str:
    return await a_vision_llm(GLOBAL_CAPTION_PROMPT, [image])


@injected
async def a_image_to_keywords(a_vision_llm: VisionLLM, /, image) -> list[str]:
    res = list(
        set(
            [
                s.strip()
                for s in (await a_vision_llm(KEYWORD_PROMPT, [image])).split(",")
            ]
        )
    )
    # remove number parsable keywords
    res = [s for s in res if not s.isnumeric()]
    # remove keywords composed of more than 3 words
    res = [s for s in res if len(s.split()) <= 3]
    return res


@injected
async def a_image_to_captions(a_vision_llm_json: VisionLLMJson, /, image) -> Captions:
    data = await a_vision_llm_json(IMAGE_PROMPT, [image])
    return Captions(**data)


@injected
async def a_image_to_headings(a_vision_llm_json: VisionLLMJson, /, image) -> Headings:
    data = await a_vision_llm_json(
        HEADINGS_PROMPT,
        [image],
        fix_context="The json should only have 'heading' and 'sub-heading' keys.",
    )
    return Headings(**data)


@injected
async def image_to_category(a_vision_llm: VisionLLM, /, image) -> str:
    prompt = """
<image>
Please specify the category of the image.
=== Example ===
Instagram, Poster, Youtube thumbnail, Book cover, ...
===============
"""
    return await a_vision_llm(prompt, [image])


@injected
async def a_vision_llm__dummy(text, images: list[Image]) -> str:
    return "dummy"


@injected
async def a_image_to_detail(
    a_image_to_global_caption,
    a_image_to_keywords,
    a_image_to_captions,
    a_image_to_headings,
    /,
    image: PIL.Image.Image,
) -> DetailV1:
    """
    a main function to perform image to detail json conversion.
    :param image:
    :return:
    """
    global_caption: str = a_image_to_global_caption(image)
    keywords: list[str] = a_image_to_keywords(image)
    captions: Captions = a_image_to_captions(image)
    headings: Headings = a_image_to_headings(image)
    global_caption, keywords, captions, headings = await asyncio.gather(
        global_caption, keywords, captions, headings
    )
    return DetailV1(
        description=global_caption,
        keywords=keywords,
        captions=captions,
        headings=headings,
    )


@injected
async def a_store_detail(
    a_image_to_detail, logger, /, image: PIL.Image.Image, path: Path
):
    if path.exists():
        logger.info(f"skipping storing detail at {path} because it already exists")
        return

    try:
        detail: DetailV1 = await a_image_to_detail(image)
        detail_dict = detail.model_dump()
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"failed to get detail for {path} with error:{e}. \n {trace}")
        detail_dict = dict(
            metadata=dict(
                status="failure",
                cause=str(e),
            )
        )
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(detail_dict, indent=4))
    logger.info(f"stored detail at {path}")


@injected
async def a_store_all_details(
    a_store_detail,
    a_map_progress,
    logger,
    /,
    image_id_pairs: Union[AsyncIterator[IdImagePath], List[IdImagePath]],
    dst_root: Path,
    total=None,
):
    """
    Stores the detail json at dst_root / id[:3] / (id + ".json") for each image_id_pair
    :param image_id_pairs:
    :param dst_root:
    :return:
    """
    dst_root = Path(dst_root)
    dst_root.mkdir(exist_ok=True, parents=True)

    async def task(t: IdImagePath):
        try:
            await a_store_detail(
                await t.aload(), dst_root / t.id[:3] / (t.id + ".json")
            )
        except JSONDecodeError as jde:
            trace = traceback.format_exc()
            logger.error(
                f"failed to store detail for {id} with JSONDecodeError:{jde}. we need to fix and retry. \n {trace}"
            )
        except BadRequestError as bre:
            trace = traceback.format_exc()
            logger.error(
                f"failed to store detail for {id} with BadRequestError:{bre}. we need to fix and retry. \n {trace}"
            )
            # I need to notify how it failed...

    async for item in a_map_progress(
        task, image_id_pairs, desc=f"storing details at {dst_root}", total=total
    ):
        pass


@injected
async def a_id_image_path_iterator_from_path(logger, /, root: Path):
    logger.info(f"iterating through {root.absolute()} for image id pairs")
    for path in tqdm(root.rglob("*"), desc=f"iterating through {root}"):
        if path.suffix.lower() in IMAGE_EXTENTIONS:
            id = path.stem
            yield IdImagePath(id, Path(path))


# for entrypoint
a_pinjected_store_all_details: Injected = a_store_all_details(
    image_id_pairs=injected("image_id_paths"),
    dst_root=injected("dst_root"),
)
# for straightforward entrypoint
a_pinjected_store_all_details_in_dir: Injected = a_store_all_details(
    image_id_pairs=a_id_image_path_iterator_from_path(injected("images_root")),
    dst_root=injected("dst_root"),
)


@injected
async def a_id_image_paths_to_dict(
    id_image_paths: AsyncIterator,
) -> dict[str, IdImagePath]:
    """
    :param id_image_paths: An asynchronous iterator containing items of type IdImagePath. Each item represents an ID and its corresponding image path.
    :return: A dictionary where the keys are the IDs and the values are the corresponding IdImagePath objects.

    Example Usage:
    ```python
    async def main():
        # Create an input iterator
        async def id_image_paths():
            yield IdImagePath(id=1, image_path="/path/to/image1.png")
            yield IdImagePath(id=2, image_path="/path/to/image2.png")
            yield IdImagePath(id=3, image_path="/path/to/image3.png")

        # Invoke the method
        result = await a_id_image_paths_to_dict(id_image_paths())

        # Print the result
        print(result)

    # Run the main function
    asyncio.run(main())
    ```

    Example Output:
    ```
    {
        1: IdImagePath(id=1, image_path="/path/to/image1.png"),
        2: IdImagePath(id=2, image_path="/path/to/image2.png"),
        3: IdImagePath(id=3, image_path="/path/to/image3.png")
    }
    ```
    """
    res = dict()
    async for item in id_image_paths:
        res[item.id] = item
    return res


@injected
async def a_id_image_iterator_from_path(
    logger, a_id_image_path_iterator_from_path, /, root: Path
):
    """
    :param root: The root directory path from which to start iterating through files.
    :return: An asynchronous iterator that yields PIL Image objects and their corresponding IDs.
    """
    logger.info(f"iterating through {root.absolute()} for image id pairs")
    async for pair in a_id_image_path_iterator_from_path(root):
        yield pair


async def aislice(aiter, start, stop):
    """
    :param aiter: An asynchronous iterable object.
    :param start: The index to start slicing from (inclusive).
    :param stop: The index to stop slicing at (exclusive).
    :return: An asynchronous generator that yields items from the given `aiter` object within the specified slice range.

    This method accepts an asynchronous iterable object `aiter`, and two integer values `start` and `stop`. It returns an asynchronous generator that yields items from `aiter`, starting
    * from the `start` index (inclusive) up to the `stop` index (exclusive).

    The `aiter` object should support the asynchronous iteration protocol, meaning it should implement the `__aiter__()` and `__anext__()` methods.

    The slicing behavior is similar to regular slicing in Python. If `start` is greater than 0, the generator will skip items until the `start` index is reached. If `stop` is less than or
    * equal to 0, the generator will stop yielding items.

    Example usage:

    ```python
    async def async_generator():
        for i in range(10):
            yield i

    ait = async_generator()
    sliced_items = [item async for item in aislice(ait, 2, 6)]
    print(sliced_items)  # Output: [2, 3, 4, 5]
    ```
    """
    async for item in aiter:
        if start > 0:
            start -= 1
            continue
        if stop <= 0:
            break
        stop -= 1
        yield item


async def alist(aiter):
    res = []
    async for item in aiter:
        res.append(item)
    return res


async def split_aiter(n_split, idx, aiter):
    """ """
    async for imid in aiter:
        imid: IdImagePath
        # _id is a hex string, so we can do modulo on it with n_batch_split.
        modulo = int(imid.id, 16) % n_split
        if modulo == idx:
            yield imid


@injected
async def a_n_split_iterator(
    opencole_n_batch_split,
    opencole_batch_index,
    /,
    id_image_aiter,
):
    """
    :param opencole_n_batch_split: The number of splits to create from the input iterator.
    :param opencole_batch_index: The index of the split to retrieve from the input iterator.
    :param id_image_aiter: The input iterator to split.
    :return: An async iterator that yields items from the specified split of the input iterator.

    """
    async for item in split_aiter(
        opencole_n_batch_split, opencole_batch_index, id_image_aiter
    ):
        yield item


@injected
async def test_n_split_contains_all(target: AsyncIterator[IdImagePath]):
    items = []
    async for item in target:
        items.append(item)

    async def gen():
        for item in items:
            yield item

    all_id_set = set([i.id for i in items])
    seen_id_set = set()
    for i in range(5):
        async for item in split_aiter(5, i, gen()):
            seen_id_set.add(item.id)
    assert all_id_set == seen_id_set
    logger.info("test_n_split_contains_all passed")


@injected
async def a_validate_detail_json_exists(
    image_id_pairs: AsyncIterator[IdImagePath],
    dst_root: Path,
):
    bar = tqdm(desc="validating detail json existence")
    failures = []
    succeeds = []
    seen_ids = []
    async for item in image_id_pairs:
        id = item.id
        dst = dst_root / id[:3] / (id + ".json")
        bar.update()
        seen_ids.append(id)
        if not dst.exists():
            logger.error(f"detail json does not exist at {dst}")
            failures.append(Failure(f"detail json does not exist at {dst}"))
        else:
            succeeds.append(dst)
    assert len(seen_ids) == len(set(seen_ids))
    logger.info(f"found {len(seen_ids)} unique ids in src image id pairs. ")
    if failures:
        logger.error(f"validation failed with {len(failures)} failures.")
        raise Exception(
            f"validation failed with {len(failures)} failures.\n({failures})"
        )
    else:
        logger.info(f"validation succeeded with {len(succeeds)} successes.")


__meta_design__ = instances()
