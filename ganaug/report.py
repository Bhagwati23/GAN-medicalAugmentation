"""Turns saved results into the report's figures and tables (PNG, Markdown and LaTeX)  (issue #7).

    python -m ganaug.report --dataset breastmnist

Outputs in results/<dataset>/:
  fig_metrics_vs_ratio.png   main result: F1 / AUC / accuracy vs % synthetic, early vs final generator, flip/shift band
  fig_confusion.png          confusion matrices (summed over seeds, row-normalised) for 0/50/100/200 %
  fig_gan_losses.png         GAN training curves
  fig_real_vs_synthetic.png  real images next to early- and final-generator images
  fig_interpolation.png      walking through latent space: smooth changes = generator learned, not memorised
  results_table.md / .tex    mean ± std over seeds, change in F1 vs real-only, Welch t-test p-value
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")  # draw to files only (no window), works on servers/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # heatmaps for confusion matrices
import torch
from scipy.stats import ttest_ind  # significance test between two groups of seeds
from torchvision.utils import make_grid

from .augment import RATIOS
from .config import CKPT_DIR, Z_DIM, results_dir
from .data import class_names, load_split
from .dcgan import generate, load_generator

LABELS = {  # setting id (used in CSV files) -> readable name (used in figures/tables)
    "real": "Real only (0%)",
    "classical": "Real + flip/shift",
    "gan_final_50": "+50% GAN (final)",
    "gan_final_100": "+100% GAN (final)",
    "gan_final_200": "+200% GAN (final)",
    "gan_early_50": "+50% GAN (early)",
    "gan_early_100": "+100% GAN (early)",
    "gan_early_200": "+200% GAN (early)",
    "gan_final_balance": "GAN class balancing",
}
FIG_METRICS = (("f1", "Macro F1"), ("auc", "ROC-AUC"), ("accuracy", "Accuracy"))


def plot_ratio_curves(summary, fid, out):  # x-axis = amount of synthetic data, one line per generator quality
    fig, axes = plt.subplots(1, len(FIG_METRICS), figsize=(7.5, 3.0))  # close to the report column width so text stays readable
    xs = [int(r * 100) for r in RATIOS]  # 0, 50, 100, 200
    for ax, (m, title) in zip(axes, FIG_METRICS):
        for tag, color in (("final", "#1f77b4"), ("early", "#d62728")):  # blue = good generator, red = early (worse) generator
            pts = [(x, k) for x, k in zip(xs, ["real"] + [f"gan_{tag}_{x}" for x in xs[1:]]) if k in summary.index]  # 0% point = real-only baseline
            if len(pts) < 2:
                continue
            mean = [summary.loc[k, f"{m}_mean"] for _, k in pts]
            std = [summary.loc[k, f"{m}_std"] for _, k in pts]
            label = f"{tag} G (FID {fid[f'generator_{tag}']['fid']:.0f})" if fid else f"{tag} G"  # legend shows each generator's FID = quality
            ax.errorbar([x for x, _ in pts], mean, yerr=std, marker="o", capsize=4, color=color, label=label)  # error bar = std over seeds
        if "classical" in summary.index:
            cm, cs = summary.loc["classical", f"{m}_mean"], summary.loc["classical", f"{m}_std"]
            ax.axhspan(cm - cs, cm + cs, color="gray", alpha=0.15)  # grey band = classical flip/shift augmentation ± std
            ax.axhline(cm, color="gray", ls="--", lw=1, label="flip/shift aug.")
        ax.set_xticks(xs)
        ax.set_xlabel("Synthetic added (% of real train)", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)  # one shared legend below, keeps the panels uncluttered
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_confusions(confusions, names, out):
    keys = [k for k in ("real", "gan_final_50", "gan_final_100", "gan_final_200") if k in confusions]
    fig, axes = plt.subplots(1, len(keys), figsize=(14, 3.4))
    short = [n.split(",")[0][:14] for n in names]
    for ax, k in zip(axes, keys):
        cm = np.sum(confusions[k], axis=0)  # add up the confusion matrices of all seeds
        cm = cm / cm.sum(axis=1, keepdims=True)  # row-normalise: each row = recall of that true class (diagonal = per-class recall)
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                    xticklabels=short, yticklabels=short)
        ax.set_title(LABELS[k], fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_gan_losses(losses, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(losses["epoch"], losses["loss_d"], label="Discriminator")  # D loss ~1.38 = 2*ln(2): D is guessing -> balanced game
    axes[0].plot(losses["epoch"], losses["loss_g"], label="Generator")
    axes[0].set_title("DCGAN losses"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(losses["epoch"], losses["d_real"], label="D(x) real")  # both curves near 0.5 and close together = healthy, no collapse
    axes[1].plot(losses["epoch"], losses["d_fake"], label="D(G(z)) fake")
    axes[1].set_title("Discriminator outputs"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_real_vs_fake(dataset, names, out, per_class=10):  # qualitative check: do synthetic images look like the real ones?
    x, y = load_split(dataset, "train")
    panels = [("Real", [x[y == c][:per_class] for c in range(len(names))])]
    for tag in ("early", "final"):
        path = CKPT_DIR / dataset / f"G_{tag}.pt"
        G = load_generator(path)
        epoch = torch.load(path, map_location="cpu")["epoch"]
        panels.append((f"DCGAN {tag} (epoch {epoch})",
                       [generate(G, torch.full((per_class,), c), seed=7 + c) for c in range(len(names))]))  # one row per class
    fig, axes = plt.subplots(len(panels), 1, figsize=(per_class * 0.75, len(panels) * (len(names) * 0.75 + 0.45)))  # panels stacked vertically, one row per class
    for ax, (title, rows) in zip(axes, panels):
        grid = make_grid(torch.cat(rows), nrow=per_class, normalize=True, value_range=(-1, 1), pad_value=1)
        ax.imshow(grid.permute(1, 2, 0))  # CHW -> HWC for matplotlib
        ax.set_title(title)
        ax.set_yticks([(i + 0.5) * grid.shape[1] / len(names) for i in range(len(names))])
        ax.set_yticklabels([n.split(",")[0][:12] for n in names], fontsize=8)
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


@torch.no_grad()
def plot_interpolation(dataset, names, out, steps=10, rows=3):
    """Linear walk between two noise vectors z0 -> z1 (same class). Smooth transitions mean G learned a continuous
    image manifold; abrupt jumps between copies of training images would indicate memorisation."""
    G = load_generator(CKPT_DIR / dataset / "G_final.pt")
    gen = torch.Generator().manual_seed(11)
    imgs, row_names = [], []
    for r in range(rows * len(names)):
        c = r % len(names)  # alternate classes per row
        z0, z1 = torch.randn(2, Z_DIM, generator=gen)  # two random points in latent space
        t = torch.linspace(0, 1, steps).unsqueeze(1)  # 0, 0.11, ..., 1
        z = (1 - t) * z0 + t * z1  # points on the straight line between them
        imgs.append(G(z, torch.full((steps,), c)))
        row_names.append(names[c].split(",")[0])
    grid = make_grid(torch.cat(imgs), nrow=steps, normalize=True, value_range=(-1, 1), pad_value=1)
    fig, ax = plt.subplots(figsize=(steps * 0.9, rows * len(names) * 0.9 + 0.6))
    ax.imshow(grid.permute(1, 2, 0))
    ax.set_yticks([(i + 0.5) * grid.shape[1] / len(row_names) for i in range(len(row_names))])
    ax.set_yticklabels(row_names, fontsize=8)
    ax.set_xticks([])
    ax.set_title("Latent-space interpolation (final generator): z0 → z1")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_tables(summary, runs, out_dir, baseline="real"):
    rows = []
    base_f1 = runs.loc[runs.setting == baseline, "f1"].to_numpy()  # F1 of every seed of the real-only baseline
    for key in summary.index:
        row = {"Setting": LABELS.get(key, key)}
        for m in ("accuracy", "precision", "recall", "f1", "auc"):
            row[m] = f"{summary.loc[key, f'{m}_mean']:.3f} ± {summary.loc[key, f'{m}_std']:.3f}"  # mean ± std over seeds
        row["ΔF1 vs real"] = f"{summary.loc[key, 'f1_mean'] - summary.loc[baseline, 'f1_mean']:+.3f}"  # + means better than real-only
        f1 = runs.loc[runs.setting == key, "f1"].to_numpy()
        row["p (F1)"] = "–" if key == baseline else f"{ttest_ind(f1, base_f1, equal_var=False).pvalue:.3f}"  # Welch t-test; p < 0.05 = difference unlikely to be seed noise
        rows.append(row)
    table = pd.DataFrame(rows).rename(columns={"accuracy": "Accuracy", "precision": "Precision",
                                               "recall": "Recall", "f1": "Macro F1", "auc": "AUC"})
    (out_dir / "results_table.md").write_text(table.to_markdown(index=False), encoding="utf-8")  # for README / GitHub
    latex = table.to_latex(index=False).replace("±", r"$\pm$").replace("Δ", r"$\Delta$").replace("%", r"\%").replace("–", "--")
    (out_dir / "results_table.tex").write_text(latex, encoding="utf-8")  # paste-ready table for the Overleaf report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="breastmnist")
    dataset = p.parse_args().dataset
    out_dir = results_dir(dataset)
    names = class_names(dataset)

    fid_path = out_dir / "fid.json"
    fid = json.loads(fid_path.read_text()) if fid_path.exists() else None  # FID is optional for plotting
    summary = pd.read_csv(out_dir / "metrics_summary.csv", index_col=0)  # written by ganaug.experiments
    runs = pd.read_csv(out_dir / "metrics_runs.csv")  # one row per (setting, seed)

    plot_ratio_curves(summary, fid, out_dir / "fig_metrics_vs_ratio.png")
    plot_confusions(json.loads((out_dir / "confusion_matrices.json").read_text()), names, out_dir / "fig_confusion.png")
    losses = out_dir / "report_run" / "gan_losses.csv"  # run 1 = the GAN run whose checkpoints produced every reported number
    plot_gan_losses(pd.read_csv(losses if losses.exists() else out_dir / "gan_losses.csv"), out_dir / "fig_gan_losses.png")  # fall back to a fresh training log
    plot_real_vs_fake(dataset, names, out_dir / "fig_real_vs_synthetic.png")
    plot_interpolation(dataset, names, out_dir / "fig_interpolation.png")
    write_tables(summary, runs, out_dir)
    print(f"figures and tables written to {out_dir}")


if __name__ == "__main__":
    main()
