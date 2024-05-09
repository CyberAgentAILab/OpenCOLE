import random
from typing import Any

import datasets as ds

from layoutlib.schema import ElementSchema, mock_element, mock_element_features
from layoutlib.util import list_of_dict_to_dict_of_list

from .base import BaseHFDSHelper
from .util import Example


class MockHFDSHelper(BaseHFDSHelper):
    """
    Handling dataset-specific operations in HFDS dataset.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def normalize(self, example: Example) -> Example:
        return example

    def denormalize(
        self,
        example: Example,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
    ) -> Example:
        return example


def mock_hfds_example(
    schema: ElementSchema, element_features: ds.Features, max_size: int = 10
) -> Example:
    n = random.randint(1, max_size)
    example = list_of_dict_to_dict_of_list(
        [mock_element(schema, element_features) for _ in range(n)]
    )
    return example


def mock_dataset(schema: ElementSchema, size: int = 10) -> ds.Dataset:
    dataset = ds.Dataset.from_list(
        [
            mock_hfds_example(
                schema=schema, element_features=mock_element_features(schema)
            )
            for _ in range(size)
        ],
        features=mock_hfds_features(schema),
    )
    return dataset


def mock_hfds_features(schema: ElementSchema) -> ds.Features:
    return ds.Features(
        {k: ds.Sequence(v) for (k, v) in mock_element_features(schema).items()}
    )
