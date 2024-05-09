import logging
import pickle
import random
from abc import ABC
from typing import Any, NamedTuple, Optional, Type

import fsspec
import numpy as np
from sklearn.cluster import KMeans

EPS_RELATIVE = 1e-3

logging.basicConfig()
logger = logging.getLogger(__name__)


def get_kmeans_cluster_center(
    key: str,
    weight_path: Optional[str] = None,
    weights: Optional[dict[str, KMeans]] = None,
) -> np.ndarray:
    """
    Given a pre-loaded weights or path to the weights,
    return the k-means cluster centers
    """
    assert weight_path is not None or weights is not None

    if weight_path is not None:
        fs, path_prefix = fsspec.core.url_to_fs(weight_path)
        logger.info(f"Load {weight_path=}")
        with fs.open(path_prefix, "rb") as f:
            weights = pickle.load(f)

    cluster_centers = np.sort(weights[key].cluster_centers_, axis=0)  # type: ignore
    cluster_centers = cluster_centers[:, 0]
    return cluster_centers


class BucketizerMockInputs(NamedTuple):
    data: np.ndarray
    num_bin: int
    vmin: float
    vmax: float


def random_bucketizer_inputs() -> BucketizerMockInputs:
    n_data = random.randint(2, 10)
    data = np.random.randn(
        n_data,
    )
    n_bit = random.randint(1, 8)
    num_bin = 2**n_bit

    return BucketizerMockInputs(
        data=data,
        num_bin=num_bin,
        vmin=data.min().item(),
        vmax=data.max().item(),
    )


class BaseBucketizer(ABC):
    """
    Interface for bucketizer to convert continuous / discrete variables
    """

    def __init__(
        self,
        num_bin: int = 128,
        vmin: float = 0.0,
        vmax: float = 1.0,
        key: str = "",
    ) -> None:
        super().__init__()
        assert vmin < vmax
        self._num_bin = num_bin
        self._vmin = vmin
        self._vmax = vmax
        self._key = key

    def encode(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def num_bin(self) -> int:
        return self._num_bin

    @property
    def vmin(self) -> float:
        return self._vmin

    @property
    def vmax(self) -> float:
        return self._vmax

    @property
    def key(self) -> str | None:
        return self._key


class LinearBucketizer(BaseBucketizer):
    """
    Uniform bucketization between vmin and vmax
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        diff = self.vmax - self.vmin
        arr = np.arange(self.num_bin + 1) / self.num_bin
        arr = arr * diff + self.vmin

        # to avoid confusion when the value is exactly on the boundary
        arr[0] -= diff * EPS_RELATIVE
        arr[-1] += diff * EPS_RELATIVE

        starts, ends = arr[:-1], arr[1:]
        self._boundaries = ends
        centers = (starts + ends) / 2.0
        self._centers = centers[:, np.newaxis]

    def encode(self, data: np.ndarray) -> np.ndarray:  # type: ignore
        assert (
            data.min() >= self.vmin and data.max() <= self.vmax
        ), f"{self._key=}, {data=}, {self.vmin=}, {self.vmax=}"
        return np.digitize(data, bins=self._boundaries, right=False)

    def decode(self, index: np.ndarray) -> np.ndarray:  # type: ignore
        assert index.ndim == 1, f"{self._key=}, {index=}"
        assert (
            index.min() >= 0 and index.max() <= self.num_bin - 1
        ), f"{self._key=}, {index=}"
        centers = self._centers[index]
        if centers.shape[-1] == 1:
            centers = centers[:, 0]
        return centers  # type: ignore


class KMeansBucketizer(BaseBucketizer):
    """
    Adaptive bucketization based on pre-computed features
    """

    def __init__(
        self,
        model: KMeans,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        model.cluster_centers_ = np.sort(model.cluster_centers_, axis=0)  # (N, C)
        self._model = model

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data[:, np.newaxis]
        return self._model.predict(data)  # type: ignore

    def encode(self, data: np.ndarray) -> np.ndarray:  # type: ignore
        if len(data) == 0:
            return data
        else:
            return self(data)  # type: ignore

    def decode(self, index: np.ndarray) -> np.ndarray:  # type: ignore
        assert index.ndim == 1, f"{self._key=}, {index=}"
        assert (
            index.min() >= 0 and index.max() <= len(self._model.cluster_centers_) - 1
        ), f"{self._key=}, {index=}, {len(self._model.cluster_centers_)=}"
        centers = self._model.cluster_centers_[index]
        if centers.shape[-1] == 1:  # one-dimensional case
            centers = centers[:, 0]
        return centers  # type: ignore


_BUCKETIZER_FACTORY = {
    "linear": LinearBucketizer,
    "kmeans": KMeansBucketizer,
}


def bucketizer_factory(name: str) -> Type:
    assert name in _BUCKETIZER_FACTORY, name
    return _BUCKETIZER_FACTORY[name]
