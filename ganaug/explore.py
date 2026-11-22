"""Dataset exploration (issue #1): split sizes, class distribution and sample images for both candidate datasets.

    python -m ganaug.explore

Why BreastMNIST is the main dataset: only 546 training images and imbalanced (147 malignant vs 399 normal/benign),
i.e. exactly the "small medical dataset" situation where extra synthetic data could help. PneumoniaMNIST (4708
training images, chest X-ray) is ~9x larger and is used as a second dataset to see whether the conclusion holds.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid  # tiles many small images into one picture

from .config import DATASETS, RESULTS_DIR
from .data import SPLITS, class_counts, class_names, load_dataset


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    stats = {}
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(11, 7), gridspec_kw={"height_ratios": [1, 1.3]})  # top: bar charts, bottom: sample grids
    for col, name in enumerate(DATASETS):
        data = load_dataset(name)
        names = class_names(name)
        stats[name] = {
            "classes": names,
            "image_shape": list(data["train"][0].shape[1:]),  # [1, 28, 28]
            **{split: {"n": len(data[split][1]), "class_counts": class_counts(data[split][1], len(names))} for split in SPLITS},
        }

        ax = axes[0, col]
        width = 0.25
        for i, split in enumerate(SPLITS):  # grouped bars: train / val / test per class
            counts = stats[name][split]["class_counts"]
            ax.bar([k + (i - 1) * width for k in range(len(names))], counts, width, label=split)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split(",")[0][:28] for n in names], fontsize=8)
        ax.set_title(f"{name}: class distribution")
        ax.legend()

        ax = axes[1, col]
        x, y = data["train"]
        per_class = 6
        tiles = []
        for c in range(len(names)):
            tiles.extend(x[y == c][:per_class])  # first 6 training images of each class
        grid = make_grid(tiles, nrow=per_class, normalize=True, value_range=(-1, 1), pad_value=1)
        ax.imshow(grid.permute(1, 2, 0))
        ax.set_title(f"{name}: one row per class")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "dataset_overview.png", dpi=180)  # used in the report's Dataset section
    (RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))  # exact split sizes and class counts
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
