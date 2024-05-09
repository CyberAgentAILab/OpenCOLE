import argparse
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import openai
import requests  # type: ignore
import torch
from diffusers import DiffusionPipeline
from PIL import Image

from opencole.inference.tester import BASESDXLTester, BASET2ITester
from opencole.inference.util import TestInput, load_cole_data

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    BASET2ITester.register_args(parser)

    parser.add_argument("--tester", type=str, choices=tester_names(), required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    logger.info(f"{args=}")

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    # get tester-specific arguments and instantiate the tester
    tester_class = tester_factory(args.tester)
    tester = tester_class(
        **{k: v for (k, v) in vars(args).items() if k not in ["tester", "output_dir"]}
    )
    inputs: list[TestInput] = load_cole_data(split_name="designerintention_v1")

    for i, input_ in enumerate(inputs):
        output_path = output_dir / f"{input_.id}.png"
        if output_path.exists():
            logger.info(f"Skipped because {output_path=} exists.")
            continue

        logger.info(f"{input_.id=} {input_.gpt_aug_prompt=}")
        output = tester(input_.gpt_aug_prompt)
        output.save(str(output_path))

        if os.environ.get("LOGLEVEL", "DEBUG") is not None and i >= 9:
            break


class SDXLTester(BASESDXLTester):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __call__(self, prompt: str) -> Image.Image:
        width, height = self._size_sampler()
        sampling_kwargs = {"width": width, "height": height, **self.sampling_kwargs}
        return self.sample(prompt, **sampling_kwargs)


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


TESTER_MAPPING = {
    "deepfloyd": DeepFloydTester,
    "sdxl": SDXLTester,
    "dalle3": DALLE3Tester,
}


def tester_factory(name: str) -> type[BASET2ITester]:
    return TESTER_MAPPING[name]


def tester_names() -> list[str]:
    return list(TESTER_MAPPING.keys())


if __name__ == "__main__":
    main()
