import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)

CSV_PATH = str(Path(__file__).parent / "cole_{split_name}_samples.csv")


class TestInput(NamedTuple):
    id: str
    prompt: str
    gpt_aug_prompt: str | None = None
    Category: str | None = None


def load_cole_data(
    split_name: str | None = None, path: str | None = None
) -> list[TestInput]:
    if split_name is None:
        assert path is not None
        input_csv_path = path
    else:
        input_csv_path = CSV_PATH.format(split_name=split_name)

    df = load_cole_data_as_df(input_csv_path=input_csv_path)
    # note: intension might be a typo, since it refers to the properties inherent to a word
    # (e.g., intension of “cat” includes being furry, having whiskers, going “miaow,” ...)
    test_inputs = []
    for _, row in df.iterrows():
        kwargs = {"id": row["id"], "prompt": row["Intension"]}
        if "Augmented Intension with GPT4" in df.columns:
            kwargs["gpt_aug_prompt"] = row["Augmented Intension with GPT4"]
        test_inputs.append(TestInput(**kwargs))
    return test_inputs


def load_cole_data_as_df(
    input_csv_path: str = CSV_PATH.format(split_name="designerintention_v1"),
) -> pd.DataFrame:
    return pd.read_csv(input_csv_path, dtype={"id": object})
