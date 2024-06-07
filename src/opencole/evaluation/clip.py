import torch
from PIL import Image
from torch import Tensor
from transformers import AutoModel, AutoProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad
def extract_text_features(
    texts: list[str], model: AutoModel, processor: AutoProcessor
) -> Tensor:
    inputs = processor.tokenizer(
        text=texts, return_tensors="pt", padding=True, truncation=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    text_features = model.get_text_features(**inputs)
    return text_features


@torch.no_grad
def extract_image_features(
    images: list[Image.Image], model: AutoModel, processor: AutoProcessor
) -> Tensor:
    inputs = processor.image_processor(images=images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    image_features = model.get_image_features(**inputs)
    return image_features


def extract_features(
    inputs: list[str] | list[Image.Image],
    ext: str,
    model: AutoModel,
    processor: AutoProcessor,
) -> Tensor:
    if ext == "png":
        return extract_image_features(inputs, model, processor)
    elif ext == "txt":
        return extract_text_features(inputs, model, processor)
    else:
        raise NotImplementedError(f"Unsupported extension: {ext}")
