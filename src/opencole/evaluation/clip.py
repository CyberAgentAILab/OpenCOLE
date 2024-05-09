from PIL import Image
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor


def extract_text_features(
    texts: list[str], model: CLIPModel, processor: CLIPProcessor
) -> Tensor:
    text_features = model.get_text_features(
        **getattr(processor, "tokenizer")(
            texts, return_tensors="pt", padding=True, truncation=True
        )
    )
    return text_features


def extract_image_features(
    images: list[Image.Image], model: CLIPModel, processor: CLIPProcessor
) -> Tensor:
    image_features = model.get_image_features(
        **getattr(processor, "image_processor")(images, return_tensors="pt")
    )
    return image_features


def extract_features(
    inputs: list[str] | list[Image.Image],
    ext: str,
    model: CLIPModel,
    processor: CLIPProcessor,
) -> Tensor:
    if ext == "png":
        return extract_image_features(inputs, model, processor)
    elif ext == "txt":
        return extract_text_features(inputs, model, processor)
