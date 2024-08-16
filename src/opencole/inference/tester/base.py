import argparse
import logging
import random
from typing import Any

import torch

logger = logging.getLogger(__name__)

NEGATIVE = "deep fried watermark cropped out-of-frame low quality low res oorly drawn bad anatomy wrong anatomy extra limb missing limb floating limbs (mutated hands and fingers)1.4 disconnected limbs mutation mutated ugly disgusting blurry amputation synthetic rendering"


class BaseTester:
    def __init__(
        self,
        seed: int | None = None,
        max_num_trials: int = 5,
        chunk_num: int | None = None,
        chunk_index: int | None = None,
    ) -> None:
        if seed is None:
            seed = random.randint(0, 2**32)
        # https://huggingface.co/docs/diffusers/en/using-diffusers/reproducibility
        if torch.cuda.is_available():
            self._generator = torch.manual_seed(seed)
        else:
            self._generator = torch.Generator(device="cpu").manual_seed(seed)

        self._max_num_trials = max_num_trials
        self._chunk_num = chunk_num
        self._chunk_index = chunk_index

    @property
    def generator(self) -> torch.Generator:
        return self._generator

    @property
    def max_num_trials(self) -> int:
        return self._max_num_trials

    @classmethod
    def register_args(cls, parser: argparse.ArgumentParser) -> None:
        """
        Register common arguments for all testers.
        """
        # control the randomness of the sampling
        parser.add_argument("--seed", type=int, default=None)

        # recovering from invalid outputs by retry
        parser.add_argument("--max_num_trials", type=int, default=5)

        # parallel execution (see BaseTester.get_chunks)
        parser.add_argument(
            "--chunk_num",
            type=int,
            default=None,
            help="used for parallel execution in combination with --chunk_index.",
        )
        parser.add_argument("--chunk_index", type=int, default=None)

    def get_chunk(self, data: list[Any]) -> list[Any]:
        """
        Used to evenly slice a list into chunks.
        If chunk_num is None, the original list is returned.
        """
        if self._chunk_num is None or self._chunk_index is None:
            logger.info("Processing all the data without chunking ...")
            return data
        else:
            assert 0 < self._chunk_num and 0 <= self._chunk_index < self._chunk_num
            indexes = range(self._chunk_index, len(data), self._chunk_num)
            chunk = [data[index] for index in indexes]
            logger.info(
                f"Returning {len(chunk)} out of {len(data)} samples. ({self._chunk_index=}, {self._chunk_num=})"
            )
            return chunk

    @property
    def use_chunking(self) -> bool:
        return self._chunk_num is not None and self._chunk_index is not None
