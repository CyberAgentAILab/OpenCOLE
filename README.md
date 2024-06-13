<div align="center">
<h1> <a href="https://arxiv.org/abs/2406.08232">OpenCOLE: Towards Reproducible Automatic Graphic Design Generation </a> </h1>

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

🤔 Automatic generation of graphic designs has recently received considerable attention.

😦 However, the state-of-the-art approaches are **complex** and rely on **proprietary** datasets, which creates reproducibility barriers.

🔥 In this paper, we propose an open framework for automatic graphic design called OpenCOLE, where we build a modified version of the pioneering [COLE [Jia+, arXiv'23]](https://graphic-design-generation.github.io/) and **train our model exclusively on publicly available datasets**.

🚀 Based on GPT4V evaluations, our model shows promising performance comparable to the original COLE. We release the pipeline and training results to encourage **open development**.

# Setup

## Requirements

- [poetry](https://python-poetry.org/)
- [direnv](https://github.com/direnv/direnv)

## Install

```
poetry install
```

## Dataset
OpenCOLE dataset (v1) is available at [`cyberagent/opencole`](https://huggingface.co/datasets/cyberagent/opencole) in HuggingFace dataset hub.

## Pre-trained models
- text_to_image: [`cyberagent/opencole-stable-diffusion-xl-base-1.0-finetune`](https://huggingface.co/cyberagent/opencole-stable-diffusion-xl-base-1.0-finetune)
- typography_lmm: [`cyberagent/opencole-typographylmm-llava-v1.5-7b-lora`](https://huggingface.co/cyberagent/opencole-typographylmm-llava-v1.5-7b-lora)

## Environment variables

Some part requires additional environment variables. We recommend to use [direnv](https://direnv.net/).
Please copy the template in [.envrc.example](.envrc.example) and modify it on your own.

```bash
cp .envrc.example .envrc
```


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
