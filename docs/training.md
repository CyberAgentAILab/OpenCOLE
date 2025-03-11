# Overview

We decide not to integrate the training code for each part due to high maintenance cost. Please refer to the original repositories. However, we provide a set of scripts to convert the OpenCOLE dataset into some dataset formats that are ready to use for the following models.

- text-to-image: [SimpleTuner](https://github.com/bghira/SimpleTuner)
- typographylmm: [LLaVA1.5](https://github.com/haotian-liu/LLaVA)

# Data Generation

## Text-to-image

We use [SimpleTuner](https://github.com/bghira/SimpleTuner) for training text-to-image part.
The following script saves the input prompt and the target image given each sample for SimpleTuner.

```python
uv run python -m opencole.preprocess.create_dataset_for_text_to_image --output_dir <OUTPUT_DIR> --max_size <MAX_SIZE>
```

## TypographyLMM

We use [LLaVA1.5](https://github.com/haotian-liu/LLaVA) for training TypographyLMM part.
```python
uv run python -m opencole.preprocess.create_dataset_for_typographylmm --output_dir <OUTPUT_DIR>
```

# Model Conversion

## Convert LLaVA1.5 model to a model compatible with transformers

1. Saving all the weights in a single file

Note: this should be done using [LLaVA](https://github.com/haotian-liu/LLaVA)'s codebase.

```python
python dump_full_llava_weight_dicts.py --model_base <MODEL_BASE> --model_base liuhaotian/llava-v1.5-7b
```

2. Conversion

For example, to convert a llava1.5-based model, please run the following.

```python
uv run python -m opencole.hf_support.convert_llava_to_hf --model_path <MODEL_PATH> --output_dir <OUTPUT_DIR> --text_model_id lmsys/vicuna-7b-v1.5 --vision_model_id openai/clip-vit-large-patch14-336
```
