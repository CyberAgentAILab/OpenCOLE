# Overview

We generate step-by-step since each phase loads billion-scale model.
Note:
- `LOGLEVEL=(ERROR|WARNING|INFO|DEBUG)` will control the log level.

# Instruction
## K-shot LLM

We use LLMs for chat and do in-context learning by retrieving a few samples, instead of a fine-tuned DesignLLM in COLE.

```python
poetry run python -m opencole.inference_launcher.k_shot_llm --input_hfds <INPUT_HFDS> --output_dir <OUTPUT_DIR>
```

If you want to test your own set of inputs, please make a csv file as follows.
Each row should contain `Intension` (make sure to use double quatation) and `id` (just for identifying each sample).
Please use `--input_intention_csv <INPUT_INTENTION_CSV>` to pass it.

```csv
id,Intension
00001,"Design an animated digital poster for a fantasy concert where Joe Hisaishi performs on a grand piano on a moonlit beach. The scene should be magical, with bioluminescent waves and a starry sky. The poster should invite viewers to an exclusive virtual event."
```

Note: if you have access to cpu, installing `faiss-cpu` might enable faster ANN search.

## Text-to-image

We directly generate a single image from text, unlike the original COLE ((i) text -> background + (ii) (text, background) -> foreground).

```python
poetry run python -m opencole.inference_launcher.text_to_image --pretrained_model_name_or_path <PRETRAINED_MODEL_NAME_OR_PATH> --detail_dir <DETAIL_DIR> --output_dir <OUTPUT_DIR>
```

Note: if you only have UNet weights, which happens in fine-tuning SDXL on your own, please use `--unet_dir <UNET_DIR> --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0` instead of `--pretrained_model_name_or_path`.

## TypographyLMM

```python
poetry run python -m opencole.inference_launcher.typographylmm --pretrained_model_name_or_path <PRETRAINED_MODEL_NAME_OR_PATH> --image_dir <IMAGE_DIR> --detail_dir <DETAIL_DIR> --output_dir <OUTPUT_DIR>
```

## Rendering

```python
poetry run python -m opencole.inference_launcher.render --image_dir <IMAGE_DIR> --typography_dir <TYPOGRAPHY_DIR> --output_dir <OUTPUT_DIR>
```

# Other Baselines

As the original [COLE](https://arxiv.org/abs/2311.16974) tested, text-to-image models with GPT-4 augmented prompts are strong baselines.
Three models (two are from HF and one is API(AzureOpenAI)) can be used by just switching `--tester`.

```
poetry run python -m opencole.inference_launcher.gpt4_augmented_baselines --tester (sdxl|deepfloyd|dalle3) --output_dir <OUTPUT_DIR>
```
