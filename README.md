<div align="center">
<h1> OpenCOLE: Towards Reproducible Automatic Graphic Design Generation </h3>

<h4 align="center">
    <a href="https://naoto0804.github.io/">Naoto Inoue</a>&emsp;
    <a href="https://scholar.google.co.jp/citations?user=ekIeOUAAAAAJ&hl=en">Kento Masui</a>&emsp;
    <a href="https://scholar.google.co.jp/citations?user=fdXoV1UAAAAJ">Wataru Shimoda</a>&emsp;
    <a href="https://sites.google.com/view/kyamagu">Kota Yamaguchi</a>&emsp;
    <br>
    CyberAgent
</h4>

<h4 align="center">
<a href="https://sites.google.com/view/gdug-workshop/">Workshop on Graphic Design Understanding and Generation</a> (at CVPR2024)
</h4>

![alt text](figs/main_results.png)

</div>

# Overview

This is an official repository for the paper.

# Setup

## Requirements

- [poetry](https://python-poetry.org/)
- [direnv](https://github.com/direnv/direnv)

## Install

```
poetry install
```

## (Dataset)
OpenCOLE dataset (v1) is available at [`cyberagent/opencole`](https://huggingface.co/datasets/cyberagent/opencole) in HuggingFace dataset hub.

## (Pre-trained models)
- text_to_image: [`cyberagent/opencole-stable-diffusion-xl-base-1.0-finetune`](https://huggingface.co/cyberagent/opencole-stable-diffusion-xl-base-1.0-finetune)
- typography_lmm: [`cyberagent/opencole-typographylmm-llava-v1.5-7b-lora`](https://huggingface.co/cyberagent/opencole-typographylmm-llava-v1.5-7b-lora)

# Inference

Please refer to [inference.md](./docs/inference.md).

# Evaluation

We provide a script for GPT4V-based evaluation on generated images.

```python
poetry run python -m opencole.evaluation.eval_gpt4v --input_dir <INPUT_DIR> --output_path <OUTPUT_PATH>
```

# Training

Please refer to [training.md](./docs/training.md).

# Citation

If you find this code useful for your research, please cite our paper:

```
@inproceedings{inoue2024opencole,
  title={{OpenCOLE: Towards Reproducible Automatic Graphic Design Generation}},
  author={Naoto Inoue and Kento Masui and Wataru Shimoda and Kota Yamaguchi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2024},
}
```

# Acknowledgement
This repository has been migrated from the internal repo. Despite the fact that commit logs are not visible, all the contributors have made significant contributions to the repository.

- [@proboscis](https://github.com/proboscis): OpenCOLE dataset construction
- [@shimoda-uec](https://github.com/shimoda-uec): TypographyLMM
- [@kyamagu](https://github.com/kyamagu): renderer
- [@naoto0804](https://github.com/naoto0804): other (bunch of) stuffs