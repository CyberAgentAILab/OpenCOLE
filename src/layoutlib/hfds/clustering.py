"""
Clustering coordinates for dataset-adaptive tokens
"""

import argparse
import logging
import os
import pickle
import time
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from . import (
    BaseHFDSHelper,
    clustering_attribute_set_factory,
    hfds_factory,
    hfds_helper_factory,
)
from .util import filter_layer, is_expected_layer_type

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 20
MAIN_KEY = "width"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hfds_name", type=str, default="crello")

    parser.add_argument("--result_dir", type=str, default="tmp/weights/clustering")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument(
        "--max_bbox_num",
        type=int,
        default=None,
        help="filter number of bboxes to avoid too much time consumption in kmeans (e.g., 10000)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="if number of elements is larger than this, ignore the layout",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    weight_path = Path(
        f"{args.result_dir}/{args.hfds_name}_max{args.max_seq_length}_kmeans_train_clusters.pkl"
    )
    if not weight_path.parent.exists():
        weight_path.parent.mkdir(parents=True, exist_ok=True)

    models = {}
    dataset_dict = hfds_factory(args.hfds_name)
    dataset = dataset_dict["train"].shuffle()
    hfds_helper = hfds_helper_factory(
        hfds_name=args.hfds_name, features=dataset.features
    )

    for key in clustering_attribute_set_factory(args.hfds_name):
        models.update(
            _fit(
                dataset=dataset,
                hfds_helper=hfds_helper,
                key=key,
                max_seq_length=args.max_seq_length,
                result_dir=args.result_dir,
                max_bbox_num=args.max_bbox_num,
            )
        )

    with open(weight_path, "wb") as f:
        pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)


def _fit(
    dataset: ds.Dataset,
    hfds_helper: BaseHFDSHelper,
    key: str,
    max_seq_length: int,
    result_dir: str,
    max_bbox_num: int | None = None,
) -> dict[str, KMeans]:
    n_clusters_list = [2**i for i in range(1, 9)]

    filter_is_text = partial(
        is_expected_layer_type,
        feature=dataset.features["type"].feature,
        layer_names={hfds_helper.text_element_type_name},
    )

    data = []
    for i, example in enumerate(tqdm(dataset, desc=f"Loading ({key=} ...)")):
        example = hfds_helper.normalize(example)
        example = filter_layer(filter_func=filter_is_text, example=example)
        N = len(example[MAIN_KEY])
        if N <= 0 or N > max_seq_length:
            continue

        data.extend(example[key])

        if max_bbox_num is not None and len(data) > max_bbox_num:
            logger.warning("Subsampling bboxes because there are too many for kmeans")
            break

        if os.environ.get("LOGLEVEL", "INFO") == "DEBUG" and i >= 9:
            break

    arr = np.array(data)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim == 2:
        pass
    else:
        raise NotImplementedError

    models = {}
    weight_path = Path(
        f"{result_dir}/{hfds_helper.name}_max{max_seq_length}_kmeans_train_clusters.pkl"
    )
    if not weight_path.parent.exists():
        weight_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(data)
    logger.info(f"Start clustering ({key=}, {n_samples=})")
    for n_clusters in tqdm(n_clusters_list, desc=f"Fitting ({key=} ...)"):
        start_time = time.time()
        kwargs = {
            "n_clusters": n_clusters,
            "n_init": "auto",
        }
        models[f"{key}-{n_clusters}"] = KMeans(**kwargs).fit(X=arr)

        time_elapsed = time.time() - start_time
        logger.debug(f"{n_clusters=}, {time_elapsed=}s")

    return models


def load_clustering_model(weight_path: str) -> dict[str, KMeans]:
    assert os.path.exists(weight_path), weight_path
    with open(weight_path, "rb") as f:
        models: dict[str, KMeans] = pickle.load(f)
    return models


def _visualize_stats() -> None:
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--hfds_name", type=str, default="crello")
    parser.add_argument("--attribute", type=str, required=True)
    parser.add_argument(
        "--max_bbox_num",
        type=int,
        default=None,
        help="filter number of bboxes to avoid too much time consumption in kmeans (e.g., 10000)",
    )
    args = parser.parse_args()

    dataset_dict = hfds_factory(args.hfds_name)
    features = dataset_dict["train"].features
    hfds_helper = hfds_helper_factory(hfds_name=args.hfds_name, features=features)

    arr = []
    for x in tqdm(dataset_dict["train"], desc=f"Loading {args.hfds_name=}"):
        x = hfds_helper.normalize(x)
        data = x[args.attribute]
        if isinstance(data, list):
            arr.extend(data)
        else:
            arr.append(data)

        if args.max_bbox_num is not None and len(data) > args.max_bbox_num:
            logger.warning("Subsampling bboxes because there are too many for kmeans")
            break

    # vmin, vmax = min(arr), max(arr)
    # offset = (vmax - vmin) * 0.1
    # plt.xlim(vmin - offset, vmax + offset)
    plt.hist(arr, bins=100)
    # plt.xlim(0.0, 0.02)
    plt.xlabel(args.attribute)
    plt.ylabel("Frequency")
    # plt.yscale("log")
    plt.title(f"{args.hfds_name} - {args.attribute}")
    plt.savefig(f"{args.hfds_name}_{args.attribute}_hist.png")


_CLUSTERING_DEFAULT_WEIGHT_PATHS = {
    "crello": Path(__file__).parent.parent
    / "weights"
    / f"crello_max{MAX_SEQ_LENGTH}_kmeans_train_clusters.pkl",
}


def clustering_default_weight_path_factory(hfds_name: str) -> str:
    return str(_CLUSTERING_DEFAULT_WEIGHT_PATHS[hfds_name])


if __name__ == "__main__":
    main()
