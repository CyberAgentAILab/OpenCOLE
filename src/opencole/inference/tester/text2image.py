import argparse
import json
import math
import os
import random
from functools import partial
from io import BytesIO
from typing import Any

import openai
import requests
import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    AuraFlowPipeline,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    FluxPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from PIL import Image

from opencole.inference.tester.base import BaseTester

NEGATIVE = "deep fried watermark cropped out-of-frame low quality low res oorly drawn bad anatomy wrong anatomy extra limb missing limb floating limbs (mutated hands and fingers)1.4 disconnected limbs mutation mutated ugly disgusting blurry amputation synthetic rendering"


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

    def __call__(self) -> dict[str, int]:
        aspect_ratio = random.choices(self._aspect_ratios, weights=self._weights, k=1)[
            0
        ]
        W, H = SizeSampler.calculate_size_by_pixel_area(aspect_ratio, self._resolution)
        return {"width": W, "height": H}

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


class BASET2ITester(BaseTester):
    # base class for all text-to-image models (e.g., proprietary models, diffusers, etc.)
    # just handles stuffs around output image size
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
        self._sampling_kwargs = {"generator": self.generator}

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

    @property
    def size_sampler(self) -> SizeSampler:
        return self._size_sampler

    @property
    def sampling_kwargs(self) -> dict[str, Any]:
        return self._sampling_kwargs


class SDXLTester(BASET2ITester):
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
        self._use_negative_prompts = use_negative_prompts

        self._sampling_kwargs["guidance_scale"] = guidance_scale
        self._sampling_kwargs["num_inference_steps"] = num_inference_steps

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

    def __call__(self, prompt: str) -> Image.Image:
        kwargs = {**self.size_sampler(), **self.sampling_kwargs}

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


class SD3Tester(BASET2ITester):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        use_compel: bool = True,
        use_negative_prompts: bool = True,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._use_compel = use_compel
        self._use_negative_prompts = use_negative_prompts

        self._sampling_kwargs["guidance_scale"] = guidance_scale
        self._sampling_kwargs["num_inference_steps"] = num_inference_steps

        pipeline_kwargs = {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "torch_dtype": torch.bfloat16,
        }

        pipeline = StableDiffusion3Pipeline.from_pretrained(**pipeline_kwargs)
        pipeline.to("cuda")

        self._pipeline = pipeline

    def __call__(self, prompt: str) -> Image.Image:
        kwargs = {**self.size_sampler(), **self.sampling_kwargs}

        if self._use_negative_prompts:
            image = self._pipeline(
                prompt=prompt, negative_prompt=NEGATIVE, **kwargs
            ).images[0]
        else:
            image = self._pipeline(prompt=prompt, **kwargs).images[0]
        return image


class FluxTester(BASET2ITester):
    # https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        enable_model_cpu_offload: bool = True,
        run_on_low_vram_gpus: bool = True,  # to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if "schnell" in pretrained_model_name_or_path:
            # https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux#timestep-distilled
            self._sampling_kwargs["guidance_scale"] = 0.0
            self._sampling_kwargs["num_inference_steps"] = 4
            self._sampling_kwargs["max_sequence_length"] = 256
        elif "dev" in pretrained_model_name_or_path:
            # https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux#guidance-distilled
            self._sampling_kwargs["guidance_scale"] = 7.5
            self._sampling_kwargs["num_inference_steps"] = 30
        else:
            raise NotImplementedError

        pipeline = FluxPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch.bfloat16
        )
        if run_on_low_vram_gpus:
            pipeline.enable_sequential_cpu_offload()
            pipeline.vae.enable_slicing()
            pipeline.vae.enable_tiling()
        else:
            if enable_model_cpu_offload:
                pipeline.enable_model_cpu_offload()
        self._pipeline = pipeline

    def __call__(self, prompt: str) -> Image.Image:
        kwargs = {**self.size_sampler(), **self.sampling_kwargs}
        # note:
        # - negative prompts are not supported in Flux
        # - longer max_sequence_length allows us to avoid using compel
        image = self._pipeline(prompt=prompt, **kwargs).images[0]
        return image


class AuraFlowTester(BASET2ITester):
    # https://huggingface.co/fal/AuraFlow
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._pipeline = AuraFlowPipeline.from_pretrained(
            "fal/AuraFlow", torch_dtype=torch.float16
        ).to("cuda")

        # note: following default parameters in the doc:
        self._sampling_kwargs["guidance_scale"] = 3.5
        self._sampling_kwargs["num_inference_steps"] = 50

    def __call__(self, prompt: str) -> Image.Image:
        kwargs = {**self.size_sampler(), **self.sampling_kwargs}
        image = self._pipeline(prompt=prompt, **kwargs).images[0]
        return image


class DALLE3Tester(BASET2ITester):
    """
    https://learn.microsoft.com/ja-jp/azure/ai-services/openai/dall-e-quickstart?tabs=dalle3%2Ccommand-line&pivots=programming-language-python
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        api_version = "2024-02-15-preview"
        self._client = openai.AzureOpenAI(
            api_version=api_version,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )

    def __call__(self, prompt: str) -> Image.Image:
        # note: sleep is not necessary for low TPM

        n_retry = 0
        while n_retry < self.max_num_trials:
            try:
                response = self._client.images.generate(
                    model=os.getenv("AZURE_OPENAI_DALLE3_DEPLOYMENT_NAME"),
                    prompt=prompt,
                    n=1,
                    size="1024x1024",  # (1024x1024、1792x1024、1024x1792)
                    # response_format="b64_json",  # url or b64_json
                    # quality="hd",  # (standard, hd)
                    # style="natural",  # (vivid or natural)
                )
            except openai.BadRequestError as e:
                print(e)
                n_retry += 1
                continue
            break

        if n_retry == self.max_num_trials:
            print(f"{n_retry=} reached the maximum retry count {self.max_num_trials=}.")
            image = Image.new("RGB", (1024, 1024), (255, 255, 255))
        else:
            json_response = json.loads(response.model_dump_json())
            image_url = json_response["data"][0][
                "url"
            ]  # extract image URL from response
            image_bytes = requests.get(image_url).content  # download the image
            image = Image.open(BytesIO(image_bytes))
        return image


class DeepFloydTester(BASET2ITester):
    """
    Source: https://github.com/deep-floyd/IF
    """

    def __init__(self, enable_model_cpu_offload: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        variant = "fp16"
        torch_dtype = torch.float16

        self.stage_1 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-XL-v1.0", variant=variant, torch_dtype=torch_dtype
        )
        if enable_model_cpu_offload:
            self.stage_1.enable_model_cpu_offload()

        self.stage_2 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            text_encoder=None,
            variant=variant,
            torch_dtype=torch_dtype,
        )
        if enable_model_cpu_offload:
            self.stage_2.enable_model_cpu_offload()

        safety_modules = {
            "feature_extractor": self.stage_1.feature_extractor,
            "safety_checker": self.stage_1.safety_checker,
            "watermarker": self.stage_1.watermarker,
        }
        self.stage_3 = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            **safety_modules,
            torch_dtype=torch_dtype,
        )
        if enable_model_cpu_offload:
            self.stage_3.enable_model_cpu_offload()

    def __call__(self, prompt: str) -> Image.Image:
        width, height = self._size_sampler()

        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(prompt)
        image = self.stage_1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=self.generator,
            output_type="pt",
            width=width // 16,
            height=height // 16,
        ).images
        image = self.stage_2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=self.generator,
            output_type="pt",
            width=width // 4,
            height=height // 4,
        ).images
        image = self.stage_3(
            prompt=prompt,
            image=image,
            generator=self.generator,
            noise_level=100,
        ).images[0]
        return image


T2I_TESTER_MAPPING = {
    "deepfloyd": DeepFloydTester,
    "sdxl": SDXLTester,
    "sd3": SD3Tester,
    "dalle3": DALLE3Tester,
    "flux1_dev": partial(
        FluxTester, pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev"
    ),
    "flux1_schnell": partial(
        FluxTester, pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell"
    ),
    "auraflow": AuraFlowTester,
}


def t2itester_factory(name: str) -> type[BASET2ITester]:
    return T2I_TESTER_MAPPING[name]


def t2i_tester_names() -> list[str]:
    return list(T2I_TESTER_MAPPING.keys())
