"""
This is a collection of transform function for layout obtained from Huggingface Dataset.
The example is a dict and expected to have the following keys.
(required)
- width
- height
- center_x (or left)
- center_y (or top)
- length: indicating the length of the other attributes.
(optional)
- type
...
"""

import random
from copy import deepcopy
from typing import Any, Callable, TypeAlias

import datasets as ds
from PIL import Image

from layoutlib.util import argsort

# this is a type alias for a data emitted from huggingface datasets
Example: TypeAlias = dict[str, Any]
# this is a type to indicate a subset of Example that can be handled by layoutlib.manager
ListOnlyExample: TypeAlias = dict[str, list[Any]]

FilterFunc = Callable[[Example, int], bool]


def _is_row_like(example: Any) -> bool:
    """
    Minimal validation on the input type.
    """
    if isinstance(example, dict):
        return True
    elif isinstance(example, ds.formatting.formatting.LazyRow):
        return True
    else:
        return False


def _find_main_key(example: Example) -> str:
    for key in [
        "left",
        "top",
        "type",
        "width",
        "height",
        "center_x",
        "center_y",
    ]:
        if key in example:
            return key
    raise ValueError(f"Cannot find main key from {example}")


def reorganize(
    example: Example,
    indexes: list[int],
    reorganizable_keys: set[str] = set(),
) -> dict:
    """
    Rearrange the order of the elements in the example.
    This function is used for many operations, such as shuffling, sorting, and filtering.
    It is mainly designed for processing crello in HFDS format,
    but it can be used for other purposes by explicitly controlling reorganizable_keys.
    """
    assert _is_row_like(example)
    assert isinstance(indexes, list)

    if len(reorganizable_keys) == 0:
        reorganizable_keys = get_reorganizable_keys(example)

    outputs = deepcopy(example)  # avoid in-place modification
    for key, value in example.items():
        if key in reorganizable_keys:
            outputs[key] = [value[ind] for ind in indexes]

    if "length" in outputs:
        outputs["length"] = len(indexes)

    return outputs


def get_reorganizable_keys(example: Example) -> set[str]:
    """
    Look for fields having the same length to rearrange the element order.
    """
    assert "length" in example and isinstance(example["length"], int), example
    keys = []
    for k, v in example.items():
        if isinstance(v, list) and len(v) == example["length"]:
            keys.append(k)
    return set(keys)


def shuffle_transform(example: Example, reorganizable_keys: set[str] = set()) -> dict:
    assert _is_row_like(example)
    main_key = _find_main_key(example)

    if (N := len(example[main_key])) == 0:
        return example
    else:
        indexes = list(range(N))
        indexes_shuffled = random.sample(indexes, N)
        return reorganize(
            example=example,
            indexes=indexes_shuffled,
            reorganizable_keys=reorganizable_keys,
        )


def sort_transform(
    example: Example,
    reverse: bool = False,
    reorganizable_keys: set[str] = set(),
    key: str = "type",
) -> dict:
    """
    Sort the elements by comparing values of the attribute called `key`.
    """
    assert _is_row_like(example)
    assert key in example

    if len(example[key]) == 0:
        return example
    else:
        indexes_sorted = argsort(example[key], reverse=reverse)
        return reorganize(
            example=example,
            indexes=indexes_sorted,
            reorganizable_keys=reorganizable_keys,
        )


def get_indexes_for_lexicographic_sort(example: Example) -> list[int]:
    """
    Sort the elements in descending order by each top-left coordinate.
    Priotize the top over the left.
    """
    assert _is_row_like(example)

    if "center_x" in example and "center_y" in example:
        # coordinate system: xywh
        left = [a - b / 2.0 for (a, b) in zip(example["center_x"], example["width"])]
        top = [a - b / 2.0 for (a, b) in zip(example["center_y"], example["height"])]
    elif "left" in example and "top" in example:
        # coordinate system: ltwh
        left = example["left"]
        top = example["top"]
    else:
        raise NotImplementedError

    _zip = zip(*sorted(enumerate(zip(top, left)), key=lambda c: c[1:]))
    indexes = list(list(_zip)[0])

    return indexes


def sort_lexicographic_transform(
    example: Example, reorganizable_keys: set[str] = set()
) -> dict:
    assert _is_row_like(example)
    main_key = _find_main_key(example)

    if len(example[main_key]) == 0:
        return example
    else:
        indexes_sorted = get_indexes_for_lexicographic_sort(example)
        return reorganize(
            example=example,
            indexes=indexes_sorted,
            reorganizable_keys=reorganizable_keys,
        )


def is_expected_layer_type(
    example: Example,
    index: int,
    feature: ds.ClassLabel,
    layer_names: str | set[str],
    key: str = "type",
) -> bool:
    """
    Check whether i-th element's attribute called `key` is in the layer_names.
    """
    if isinstance(layer_names, str):
        layer_names = set([layer_names])
    layer_ids = set([feature.str2int(name) for name in layer_names])
    return bool(example[key][index] in layer_ids)


def filter_layer(
    example: Example,
    filter_func: FilterFunc | list[FilterFunc],
    reorganizable_keys: set[str] = set(),
) -> dict:
    """
    Filter layers by the given function and make a new example.
    """
    if callable(filter_func):
        filter_func = [filter_func]

    indexes_filtered = []
    for i in range(example["length"]):
        if all([f(example, i) for f in filter_func]):
            indexes_filtered.append(i)

    example_filtered = reorganize(
        example=example,
        indexes=indexes_filtered,
        reorganizable_keys=reorganizable_keys,
    )
    example_filtered["length"] = len(example_filtered["type"])
    return example_filtered


def extract_class_label_mappings(
    features: ds.Features,
) -> dict[str, ds.ClassLabel]:
    class_label_mappings = {}
    for key in features.keys():
        if hasattr(features[key], "feature"):
            class_label = features[key].feature
            if isinstance(class_label, ds.ClassLabel):
                class_label_mappings[key] = class_label
    return class_label_mappings


def get_default_value(
    feature: Any, default_image_size: tuple[int, int] = (256, 256)
) -> Any:
    if isinstance(feature, ds.ClassLabel):
        return 0
    elif feature.dtype == "float32":
        return 0.0
    elif feature.dtype == "int32":
        return 0
    elif feature.dtype == "int64":
        return 0
    elif feature.dtype == "string":
        return ""
    elif isinstance(feature, ds.Sequence):
        assert feature.length != -1
        return [get_default_value(feature.feature)] * feature.length
    elif isinstance(feature, ds.Image):
        return Image.new("RGBA", default_image_size)
    else:
        raise NotImplementedError(f"Default value not fround for {feature=}")


def fill_missing_values(example: Example, features: ds.Features) -> Example:
    example_new = deepcopy(example)
    for key, feature in features.items():
        if isinstance(feature, ds.Sequence):
            if key not in example_new:
                example_new[key] = [get_default_value(feature.feature)]
        else:
            if key not in example_new:
                example_new[key] = get_default_value(feature)
    return example_new
