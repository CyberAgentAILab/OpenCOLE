import argparse
import json
import logging
import random
import time
from copy import deepcopy
from typing import Any

import torch
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from PIL import Image

from layoutlib.hfds.base import BaseHFDSHelper
from layoutlib.hfds.util import Example
from layoutlib.manager import LayoutManager
from opencole.inference.tester.base import BaseTester
from opencole.model.llava import (
    Conversation,
    SeparatorStyle,
    llava_output_validator_factory,
    load_llava,
    padding_layout_transform,
)
from opencole.schema import DetailV1, _BaseModel
from opencole.util import TYPOGRAPHYLMM_INSTRUCTION

logger = logging.getLogger(__name__)


class BaseTransformersLMTester(BaseTester):
    """
    Base class for testers that use Huggingface transformers' LMs.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float | None = None,
        num_beams: int = 1,
        max_new_tokens: int = 1024,
        # load_in_4bit: bool = False,
        # load_in_8bit: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._temperature = temperature
        self._top_p = top_p
        self._num_beams = num_beams
        self._max_new_tokens = max_new_tokens
        # self._load_in_4bit = load_in_4bit
        # self._load_in_8bit = load_in_8bit
        # assert not (load_in_4bit and load_in_8bit), "Cannot load both 4bit and 8bit models"

    @classmethod
    def register_args(cls, parser: argparse.ArgumentParser) -> None:
        super().register_args(parser)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--top_p", type=float, default=None)
        parser.add_argument("--num_beams", type=int, default=1)
        parser.add_argument("--max_new_tokens", type=int, default=1024)
        # parser.add_argument("--load_in_4bit", action="store_true")
        # parser.add_argument("--load_in_8bit", action="store_true")

    @property
    def sampling_kwargs(self) -> dict[str, Any]:
        return {
            "temperature": self._temperature,
            "do_sample": True if self._temperature > 0 else False,
            "top_p": self._top_p,
            "num_beams": self._num_beams,
            "max_new_tokens": self._max_new_tokens,
            "use_cache": True,
        }

    # def quantization_config(self, torch_dtype: torch.dtype) -> dict[str, Any] | None:
    #     if self._load_in_4bit:
    #         return BitsAndBytesConfig(
    #             load_in_4bit=True,
    #             bnb_4bit_quant_type="nf4",
    #             bnb_4bit_use_double_quant=True,
    #             bnb_4bit_compute_dtype=torch_dtype
    #         )
    #     elif self._load_in_8bit:
    #         return BitsAndBytesConfig(
    #             load_in_8bit=True,
    #         )
    #     else:
    #         return None


class TypographyLMMTester(BaseTransformersLMTester):
    def __init__(
        self,
        layout_parser: PydanticOutputParser,
        layout_manager: LayoutManager,
        hfds_helper: BaseHFDSHelper,
        pretrained_model_name_or_path: str,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.layout_parser = layout_parser
        self.layout_manager = layout_manager
        self.hfds_helper = hfds_helper

        self.model, self.processor = load_llava(
            pretrained_model_name_or_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
        self.conv = Conversation(
            system="A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=["USER", "ASSISTANT"],
            version="v1",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
        self.validator = llava_output_validator_factory("text_placement")
        self.format_instructions = self.layout_parser.get_format_instructions()
        self.instruction = f"{TYPOGRAPHYLMM_INSTRUCTION} {self.format_instructions}"

    def __call__(self, background: Image.Image, detail: DetailV1) -> Example | None:
        conv: Conversation = deepcopy(self.conv)
        context = detail.serialize_tlmm_context()
        input_ = detail.serialize_tlmm_input()
        qs = f"<image>\n{self.instruction} Context: {context} Input: {input_}"

        conv.append_message(conv.roles[0], qs)  # type: ignore
        conv.append_message(conv.roles[1], None)  # type: ignore
        prompt = conv.get_prompt()  # type: ignore
        logger.info(f"{prompt=}")

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        inputs = self.processor(text=prompt, images=background, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(
            self.model.device, self.model.dtype
        )
        inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.model.device)

        n_trial, layout = 0, None
        while n_trial < self.max_num_trials:
            if n_trial >= 1:
                logger.warning(f"Retry {n_trial} times")
            n_trial += 1
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **self.sampling_kwargs)

            # cut off the input part
            input_token_len = inputs["input_ids"].size(1)
            outputs_str = self.processor.tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]

            outputs_str = outputs_str.strip()
            if outputs_str.endswith(stop_str):
                outputs_str = outputs_str[: -len(stop_str)]  # type: ignore
            outputs_str = outputs_str.strip()

            # workaround (for bool type)
            outputs_str = outputs_str.replace("False", "false")
            outputs_str = outputs_str.replace("True", "true")

            try:
                layout = self.layout_parser.parse(outputs_str)
            except OutputParserException as e:
                logger.error(
                    f"OutputParser is not able to parse {outputs_str=} because of {e=}"
                )
                continue

            try:
                self.validator(
                    layout=layout,
                    prompt=prompt,
                )
            except Exception as e:
                logger.error(
                    f"Failed to pass {self.validator=} for {outputs_str=} because of {e=}"
                )
                continue

            break

        if layout is None:
            logger.warning("Fill the prediction by dummy objects")
            schema = self.layout_manager.schema
            elements = []
            for text in json.loads(detail.serialize_tlmm_input(shuffle=True)):
                element: dict[str, Any] = {}
                for key in self.layout_manager.schema.attribute_order:
                    if key == "text":
                        element[key] = text
                    elif key in schema.categorical_keys:
                        names = self.layout_manager.class_label_mappings[key].names
                        element[key] = random.choice(names)
                    elif key in schema.numerical_keys:
                        attr = self.layout_manager.schema.numerical_attributes[key]
                        element[key] = random.randint(0, attr.num_bin - 1)
                    else:
                        element[key] = ""
                elements.append(element)
            layout = self.layout_parser.pydantic_object(elements=elements)

        if len(layout.elements) == 0:
            logger.warning(
                "No element is generated, probably because there is no text to be placed."
            )
            return None

        canvas_width, canvas_height = background.size
        layout_hfds = self.layout_manager.layout_instance_to_hfds(layout)
        layout_hfds = self.hfds_helper.denormalize(
            layout_hfds, canvas_width=canvas_width, canvas_height=canvas_height
        )
        layout_hfds = padding_layout_transform(
            layout_hfds,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            method="from_square",
        )
        return layout_hfds


class LangChainTester(BaseTester):
    def __init__(
        self,
        chain,
        pydantic_object: type[_BaseModel],
        sleep_sec: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.chain = chain
        self.sleep_sec = sleep_sec
        self.pydantic_object = pydantic_object

    def __call__(self, input_: dict) -> _BaseModel:
        n_retry = 0
        while n_retry < self.max_num_trials:
            time.sleep(self.sleep_sec)

            try:
                output = self.chain.invoke(input_)
            except OutputParserException:
                n_retry += 1
                logger.info(f"Failed to parse, {n_retry=}-th retry")
                continue
            except Exception as e:
                n_retry += 1
                logger.info(f"Failed to parse, {n_retry=}-th retry because of {e}")
                continue
            break

        if n_retry >= self.max_num_trials:
            # when if fails to generate and parse, return a mock object
            logger.info(
                f"Failed to generate inspite of trying {self.max_num_trials=} times. Returning a mock object."
            )
            output = self.pydantic_object.get_mock()

        return output
