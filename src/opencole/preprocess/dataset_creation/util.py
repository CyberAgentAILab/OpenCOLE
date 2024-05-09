import platform

import torch
from loguru import logger
from pinjected import instance


@instance
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif platform.platform().startswith("Darwin"):
        return torch.device("mps")
    else:
        logger.warning("No GPU available, using CPU")
        return torch.device("cpu")
