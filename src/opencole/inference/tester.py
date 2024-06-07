import argparse
import json
import logging
import math
import random
import time
from copy import deepcopy
from typing import Any

import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from PIL import Image

from layoutlib.hfds.base import BaseHFDSHelper
from layoutlib.hfds.util import Example
from layoutlib.manager import LayoutManager
from opencole.model.llava import (
    Conversation,
    SeparatorStyle,
    llava_output_validator_factory,
    load_llava,
    padding_layout_transform,
)
from opencole.schema import ChildOfDetail, DetailV1, _BaseModel
from opencole.util import TYPOGRAPHYLMM_INSTRUCTION

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


class BASET2ITester(BaseTester):
    def __init__(
        self,
        image_type: str = "square",
        resolution: float = 1.0,
        resolution_type: str = "area",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._size_sampler = SizeSampler(
            image_type=image_type,
            resolution=resolution,
            resolution_type=resolution_type,
        )

    @classmethod
    def register_args(cls, parser: argparse.ArgumentParser) -> None:
        super().register_args(parser)
        parser.add_argument(
            "--image_type", type=str, default="square", choices=["arbitrary", "square"]
        )
        parser.add_argument("--resolution", type=float, default=1.0)
        parser.add_argument(
            "--resolution_type", type=str, default="area", choices=["area", "pixel"]
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @property
    def size_sampler(self) -> "SizeSampler":
        return self._size_sampler


class BASESDXLTester(BASET2ITester):
    def __init__(
        self,
        finetuned_unet_dir: str | None = None,  # for full fine-tuning
        lora_weights_path: str | None = None,  # for lora fine-tuning
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
        scheduler: str = "ddpm",
        use_compel: bool = True,
        use_negative_prompts: bool = True,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._use_compel = use_compel
        self._guidance_scale = guidance_scale
        self._num_inference_steps = num_inference_steps
        self._use_negative_prompts = use_negative_prompts

        pipeline_kwargs = {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "torch_dtype": torch.bfloat16,
        }

        assert not (
            finetuned_unet_dir is not None and lora_weights_path is not None
        ), "Two models cannot be loaded at the same time"

        if finetuned_unet_dir is not None:
            pipeline_kwargs["unet"] = UNet2DConditionModel.from_pretrained(
                finetuned_unet_dir, torch_dtype=torch.bfloat16
            )
        pipeline = StableDiffusionXLPipeline.from_pretrained(**pipeline_kwargs)
        if lora_weights_path is not None:
            pipeline.load_lora_weights(lora_weights_path)

        if scheduler == "ddpm":
            pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        elif scheduler == "dpm++2mkaras":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
        else:
            raise NotImplementedError

        compel = None
        if self._use_compel:
            compel = Compel(
                tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )

        pipeline.to("cuda")

        self._pipeline = pipeline
        self._compel = compel

        if self._use_compel and self._use_negative_prompts:
            self._negative_conditioning, self._negative_pooled = compel(NEGATIVE)  # type: ignore

    def sample(self, prompt: str, **kwargs: Any) -> Image.Image:
        if self._use_compel:
            conditioning, pooled = self._compel(prompt)  # type: ignore
            if self._use_negative_prompts:
                image = self._pipeline(
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=self._negative_conditioning,
                    negative_pooled_prompt_embeds=self._negative_pooled,
                    **kwargs,
                ).images[0]
            else:
                image = self._pipeline(
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    **kwargs,
                ).images[0]
        else:
            if self._use_negative_prompts:
                image = self._pipeline(
                    prompt=prompt, negative_prompt=NEGATIVE, **kwargs
                ).images[0]
            else:
                image = self._pipeline(prompt=prompt, **kwargs).images[0]
        return image

    @property
    def sampling_kwargs(self) -> dict[str, Any]:
        return {
            "guidance_scale": self._guidance_scale,
            "num_inference_steps": self._num_inference_steps,
            "generator": self.generator,
        }


class T2ITester(BASESDXLTester):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __call__(self, detail: ChildOfDetail) -> Image.Image:
        prompt = detail.serialize_t2i_input()
        width, height = self._size_sampler()
        sampling_kwargs = {"width": width, "height": height, **self.sampling_kwargs}
        return self.sample(prompt, **sampling_kwargs)


class SizeSampler:
    aspect_ratios: list[float] = [
        1 / 1,
        16 / 9,
        9 / 16,
    ]

    def __init__(
        self,
        image_type: str = "arbitrary",
        resolution: float = 1.0,
        resolution_type: str = "area",
        weights: list[float] | None = None,
    ) -> None:
        assert image_type in ["arbitrary", "square"], image_type
        assert resolution_type in ["area", "pixel"], resolution_type
        assert resolution > 0.0, resolution

        if weights is not None:
            assert isinstance(weights, list) and len(weights) == len(
                SizeSampler.aspect_ratios
            ), weights
        self._weights = weights

        if image_type == "arbitrary":
            self._aspect_ratios = SizeSampler.aspect_ratios
        elif image_type == "square":
            self._aspect_ratios = [
                1 / 1,
            ]
        else:
            raise NotImplementedError
        self._image_type = image_type

        self._resolution = resolution
        self._resolution_type = resolution_type

    def __call__(self) -> tuple[int, int]:
        aspect_ratio = random.choices(self._aspect_ratios, weights=self._weights, k=1)[
            0
        ]
        W, H = SizeSampler.calculate_size_by_pixel_area(aspect_ratio, self._resolution)
        return W, H

    @staticmethod
    def calculate_size_by_pixel_area(
        aspect_ratio: float, megapixels: float
    ) -> tuple[int, int]:
        """
        https://github.com/bghira/SimpleTuner/blob/main/helpers/multiaspect/image.py#L359-L371
        """
        assert aspect_ratio > 0.0, aspect_ratio
        pixels = int(megapixels * (1024**2))

        W_new = int(round(math.sqrt(pixels * aspect_ratio)))
        H_new = int(round(math.sqrt(pixels / aspect_ratio)))

        W_new = SizeSampler.round_to_nearest_multiple(W_new, 64)
        H_new = SizeSampler.round_to_nearest_multiple(H_new, 64)

        return W_new, H_new

    @staticmethod
    def round_to_nearest_multiple(value: int, multiple: int) -> int:
        """
        Round a value to the nearest multiple.
        https://github.com/bghira/SimpleTuner/blob/main/helpers/multiaspect/image.py#L264-L268
        """
        rounded = round(value / multiple) * multiple
        return max(rounded, multiple)  # Ensure it's at least the value of 'multiple'


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
