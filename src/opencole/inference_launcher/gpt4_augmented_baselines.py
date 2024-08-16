import argparse
import logging
import os
from pathlib import Path

from opencole.inference.tester.text2image import (
    BASET2ITester,
    t2i_tester_names,
    t2itester_factory,
)
from opencole.inference.util import TestInput, load_cole_data

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    BASET2ITester.register_args(parser)

    parser.add_argument("--tester", type=str, choices=t2i_tester_names(), required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--first_n", type=int, default=None)
    args = parser.parse_args()
    logger.info(f"{args=}")

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    # get tester-specific arguments and instantiate the tester
    tester_class = t2itester_factory(args.tester)
    tester = tester_class(
        **{
            k: v
            for (k, v) in vars(args).items()
            if k not in ["tester", "output_dir", "first_n"]
        }
    )
    inputs: list[TestInput] = load_cole_data(split_name="designerintention_v1")

    if args.first_n is not None:
        inputs = inputs[: args.first_n]

    for i, input_ in enumerate(inputs):
        output_path = output_dir / f"{input_.id}.png"
        if output_path.exists():
            logger.info(f"Skipped because {output_path=} exists.")
            continue

        logger.info(f"{input_.id=} {input_.gpt_aug_prompt=}")
        output = tester(input_.gpt_aug_prompt)
        output.save(str(output_path))


if __name__ == "__main__":
    main()
