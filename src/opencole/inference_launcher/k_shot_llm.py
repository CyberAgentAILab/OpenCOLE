import argparse
import json
import logging
import os
import time
from pathlib import Path

import datasets as ds
import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from opencole.inference.langchain_helper import Example, setup_model, setup_prompt
from opencole.inference.util import load_cole_data
from opencole.schema import DetailV1, IntentionV1

MAX_RETRY = 5

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument(
        "--input_hfds",
        type=str,
        required=True,
        help="a HuggingFace dataset used for in-context learning",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_intention_csv",
        type=str,
        default=None,
    )

    # model type
    parser.add_argument(
        "--model_id", type=str, default=None, help="model to use (trasnformers)"
    )
    parser.add_argument(
        "--repo_id", type=str, default=None, help="model to use (huggingface hub)"
    )
    parser.add_argument(
        "--azure_openai_model_name",
        type=str,
        default="gpt-35-turbo",
        help="model to use (Azure OpenAI)",
        choices=[
            None,
            "gpt-35-turbo",
        ],
    )
    parser.add_argument("--sleep_sec", type=float, default=3.0)

    # settings for sampling
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--max_n_retry", type=int, default=5)

    # in-context learning
    parser.add_argument("--k", type=int, default=5, help="k-shot in-context learning")
    parser.add_argument(
        "--few_shot_by",
        type=str,
        default="random",
        choices=["similarity", "random", "fixed"],
        help="similarity metric for few-shot learning",
    )
    parser.add_argument("--embeddings_cache_name", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.k > 0:
        examples_retrieval = _load_examples(
            input_hfds=args.input_hfds,
            split="train",
        )
    else:
        examples_retrieval = []
    output_parser = PydanticOutputParser(pydantic_object=DetailV1)  # type: ignore
    format_instructions = output_parser.get_format_instructions()
    model = setup_model(
        model_id=args.model_id,
        repo_id=args.repo_id,
        azure_openai_model_name=args.azure_openai_model_name,
    )
    prompt = setup_prompt(
        examples=examples_retrieval,
        format_instructions=format_instructions,
        few_shot_by=args.few_shot_by,
        k=args.k,
        embeddings_cache_name=args.embeddings_cache_name,
    )

    chain = prompt | model | output_parser

    # examples_test = _load_examples(input_hfds=args.input_hfds, split="test")
    if args.input_intention_csv is not None:
        df = pd.read_csv(args.input_intention_csv, dtype={"id": object})
        assert set(df.columns.values.tolist()) == {"Intension", "id"}
        examples_test = [
            Example(
                intention=IntentionV1(content=row["Intension"]).json(),
                id=row["id"],
            )
            for _, row in df.iterrows()
        ]
    else:
        #  If not set, designer intention v1 dataset will be used.
        examples_test = [
            Example(intention=IntentionV1(content=x.prompt).json(), id=x.id)
            for x in load_cole_data(split_name="designerintention_v1")
        ]

    for i, example in enumerate(examples_test):
        if args.azure_openai_model_name is not None:
            time.sleep(args.sleep_sec)

        n_retry = 0
        logger.info(f"Processing {example.id=} {example.intention=}...")
        while n_retry < args.max_n_retry:
            try:
                detail: DetailV1 = chain.invoke({"intention": example.intention})
            except OutputParserException:
                n_retry += 1
                logger.info(f"Failed to parse, {n_retry=}-th retry")
                if args.azure_openai_model_name is not None:
                    time.sleep(args.sleep_sec)
                continue
            except Exception as e:
                n_retry += 1
                logger.info(f"Failed to parse, {n_retry=}-th retry because of {e}")
                if args.azure_openai_model_name is not None:
                    time.sleep(args.sleep_sec)
                continue
            break

        if n_retry >= args.max_n_retry:
            # when if fails to generate and parse, return a mock object
            logger.info(
                f"Failed to generate after {args.max_n_retry=}. Returning a mock object."
            )
            detail = DetailV1.get_mock()

        output_path = output_dir / f"{example.id}.json"
        with output_path.open("w") as f:
            json.dump(detail.dict(), f, indent=4)

        if os.environ.get("LOGLEVEL", "INFO") == "DEBUG" and i == 9:
            break


def _load_examples(
    input_hfds: str, split: str = "train"
) -> list[Example]:  # key: intention, detail (as string)
    dataset_dict = ds.load_dataset(input_hfds)
    if os.environ.get("LOGLEVEL", "INFO") == "DEBUG":
        dataset_dict["train"] = dataset_dict["train"].select(range(1000))

    outputs = []
    for example in dataset_dict[split]:
        outputs.append(
            Example(
                intention=IntentionV1(content=example["intention"]).json(),
                detail=DetailV1.from_hfds(example).json(),
                id=None,
            )
        )
    return outputs  # type: ignore


if __name__ == "__main__":
    main()
