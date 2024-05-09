import random
from pathlib import Path

import numpy as np
import torch


def add_sub_directory(root: Path | str, sub: str) -> Path:
    if isinstance(root, str):
        root = Path(root)
    directory = root / sub
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_crello_image_path(
    root: Path | str, id_: str, ext: str = "png", use_subdir: bool = True
) -> Path:
    if isinstance(root, str):
        root = Path(root)

    if use_subdir:
        base_dir = root / id_[:3]
    else:
        base_dir = root

    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
    image_path = base_dir / f"{id_}.{ext}"
    return image_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
