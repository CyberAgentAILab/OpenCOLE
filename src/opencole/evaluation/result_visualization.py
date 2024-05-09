# see https://python-graph-gallery.com/391-radar-chart-with-several-individuals/

import argparse
import logging
import os
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opencole.inference.util import load_cole_data_as_df

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

plt.rcParams["font.size"] = 16

METRICS = [
    "design_and_layout",
    "content_relevance_and_effectiveness",
    "typography_and_color_scheme",
    "graphics_and_images",
    "innovation_and_originality",
]

MODEL_NAMES = (
    "gpt4_deepfloyd",
    "gpt4_sdxl",
    "gpt4_dalle3",
    "cole",
    "opencole",
)
MODEL_LABEL_MAPPINGS_SHORT = {
    "gpt4_dalle3": "Dalle3",
    "gpt4_deepfloyd": "DeepFloydIF",
    "gpt4_sdxl": "SDXL",
    "opencole": "OpenCOLE",
    "cole": "COLE",
}
COLOR_MAPPING = {
    "gpt4_dalle3": "green",
    "gpt4_deepfloyd": "blue",
    "gpt4_sdxl": "yellow",
    "opencole": "red",
    "cole": "magenta",
}
METRIC_LABEL_MAPPPING = {
    "design_and_layout": "Design\nLayout",
    "content_relevance_and_effectiveness": "Content\nRelevance",
    "typography_and_color_scheme": "Typography\nColor",
    "graphics_and_images": "Graphic\nImages",
    "innovation_and_originality": "Innovation",
}
METRIC_LABEL_MAPPPING_SHORT = {
    "design_and_layout": "(i)",
    "content_relevance_and_effectiveness": "(ii)",
    "typography_and_color_scheme": "(iii)",
    "graphics_and_images": "(iv)",
    "innovation_and_originality": "(v)",
}
CATEGORIES = {"Advertising", "Covers_Headers", "Events", "MarketingMaterials", "Posts"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument(
        "--target",
        choices=["all", "each"],
        default="all",
        help="each: category-wise comparison",
    )
    parser.add_argument(
        "--output_type",
        choices=["table", "radar", "box", "violin", "heatmap"],
        default="table",
    )
    parser.add_argument("--output_ext", type=str, default="pdf", choices=["pdf", "png"])
    args = parser.parse_args()
    logger.info(f"{args=}")

    metadata_df = load_cole_data_as_df()
    if args.target == "all":
        target_idsets_mapping = {None: set(metadata_df.id.tolist())}
    elif args.target == "each":
        target_idsets_mapping = {
            key: set(metadata_df[metadata_df["Category"] == key]["id"].tolist())  # type: ignore
            for key in CATEGORIES
        }
    else:
        raise NotImplementedError

    model_names = list(MODEL_NAMES)

    df_mapping = {}
    for model_name in model_names:
        csv_path = Path(args.input_dir) / f"{model_name}.csv"
        df = pd.read_csv(str(csv_path), dtype={"id": object})
        df = df.replace(-1, 1)  # -1 means failed to evaluate, thus give the lowest
        df_mapping[model_name] = df

    for category, target_idset in target_idsets_mapping.items():
        dfs = {}
        for model_name, base_df in df_mapping.items():
            dfs[model_name] = base_df[base_df["id"].isin(target_idset)]  # type: ignore

        prefix = f"{category}_" if category is not None else ""

        if args.output_type == "radar":
            dump_radar_chart(dfs, f"{prefix}radar.{args.output_ext}")
        elif args.output_type == "table":
            dump_table(dfs)
        elif args.output_type == "box":
            dump_plot(
                dfs=dfs, filename=f"{prefix}box.{args.output_ext}", plot_type="box"
            )
        elif args.output_type == "violin":
            dump_plot(
                dfs=dfs,
                filename=f"{prefix}violin.{args.output_ext}",
                plot_type="violin",
            )
        elif args.output_type == "heatmap":
            dump_heatmap(dfs=dfs, filename=f"{prefix}heatmap.{args.output_ext}")
        else:
            raise NotImplementedError


def dump_table(dfs: dict[str, pd.DataFrame]) -> None:
    text = ""
    for model_name, df in dfs.items():
        values = [df.loc[:, m].mean().item() for m in METRICS]
        mean = sum(values) / len(values)
        values = values + [str(mean)[:3]]
        values_str = [str(v)[:3] for v in values]
        text += f"\t{model_name} & " + " & ".join(values_str) + " \\\\\n"
    print(text)


def dump_plot(
    dfs: dict[str, pd.DataFrame], filename: str, plot_type: str = "violin"
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=len(METRICS), figsize=(18, 4))
    labels = [MODEL_LABEL_MAPPINGS_SHORT[key] for key in dfs.keys()]
    for i, metric in enumerate(METRICS):
        data = [
            df.loc[:, metric].tolist() for df in list(dfs.values())[::-1]
        ]  # revese order
        if plot_type == "box":
            axes[i].boxplot(
                data,
                labels=labels,
                showfliers=False,
            )  # will be used to label x-ticks
        elif plot_type == "violin":
            axes[i].violinplot(data, showmeans=True, showmedians=False, vert=False)
            axes[i].set_yticks(np.arange(1, len(labels) + 1), labels=labels[::-1])

        axes[i].set_xlim(0, 10)
        if i != 0:
            axes[i].get_yaxis().set_ticks([])

        axes[i].set_title(METRIC_LABEL_MAPPPING[metric])

    plt.subplots_adjust(bottom=0.2)
    fig.supxlabel("GPT4V Scores")
    plt.savefig(filename, bbox_inches="tight")


def dump_heatmap(dfs: dict[str, pd.DataFrame], filename: str) -> None:
    """
    Plot correlation between metrics
    """
    dfs = {
        k: v.drop(
            columns=[
                "id",
            ]
        )
        for k, v in dfs.items()
    }
    fig, axes = plt.subplots(nrows=1, ncols=len(dfs), figsize=(18, 4))
    for i, (name, df) in enumerate(dfs.items()):
        corr = df.corr().values
        im = axes[i].imshow(corr, vmin=0, vmax=1)
        axes[i].get_xaxis().set_ticks([])
        axes[i].get_yaxis().set_ticks([])
        axes[i].set_title(name)

    fig.subplots_adjust(right=0.96)
    cbar_ax = fig.add_axes([0.975, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.supxlabel("Correlation between metrics")
    plt.savefig(filename, bbox_inches="tight")


def dump_radar_chart(dfs: dict[str, pd.DataFrame], filename: str) -> None:
    N = len(METRICS)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # xlabel
    plt.xticks(angles[:-1], [METRIC_LABEL_MAPPPING[m] for m in METRICS])
    # ylabel
    ax.set_rlabel_position(0)
    yvalues = [(i + 1) * 2 for i in range(5)]
    plt.yticks(yvalues, [str(v) for v in yvalues], color="grey", size=7)
    plt.ylim(0, 10)

    stems = []
    for model_name, df in dfs.items():
        stems.append(model_name)
        values = [df.loc[:, m].mean().item() for m in METRICS]
        values += values[:1]
        ax.plot(
            angles,
            values,
            COLOR_MAPPING[model_name],
            linewidth=1,
            linestyle="solid",
            label=model_name,
        )
        ax.fill(angles, values, COLOR_MAPPING[model_name], alpha=0.1)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.2))
    plt.savefig(filename)


if __name__ == "__main__":
    main()
