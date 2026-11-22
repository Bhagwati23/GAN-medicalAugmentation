"""Runs every training-set setting over several seeds and evaluates on the same held-out real test set (issue #7).

    python -m ganaug.experiments --dataset breastmnist --seeds 0 1 2 3 4

Settings (9 per dataset):
  real               real training images only (baseline, 0% synthetic)
  classical          real + random flip/shift (standard augmentation; is a GAN better than this cheap trick?)
  gan_final_50/100/200   real + 50/100/200% synthetic from the FINAL generator (good quality, low FID)  -> QUANTITY axis
  gan_early_50/100/200   same amounts from the EARLY generator (poor quality, high FID)               -> QUALITY axis
  gan_final_balance  synthetic minority-class (malignant / normal) images only, until both classes are equal size
Only the TRAINING data changes between settings: same CNN, same hyper-parameters, same validation and test sets.
"""
import argparse
import json
import time

import pandas as pd

from .augment import RATIOS, SyntheticPool, build_training_set, summarize, synthetic_counts
from .classifier import evaluate, train_classifier
from .config import CKPT_DIR, results_dir
from .data import class_names, load_dataset
from .dcgan import load_generator

METRICS = ("accuracy", "precision", "recall", "f1", "auc")


def settings(n_classes, y_train):
    """(name, generator tag or None, per-class synthetic counts, classical augmentation)."""
    out = [("real", None, None, False), ("classical", None, None, True)]  # no generator for these two
    for tag in ("final", "early"):  # generator quality
        for r in RATIOS[1:]:  # 0.5, 1.0, 2.0 (0.0 is the "real" setting)
            out.append((f"gan_{tag}_{int(r * 100)}", tag, synthetic_counts(y_train, n_classes, r), False))
    out.append(("gan_final_balance", "final", synthetic_counts(y_train, n_classes, 0, mode="balance"), False))
    return out


def run(dataset, seeds, epochs, only=None):
    data = load_dataset(dataset)  # official splits
    x_train, y_train = data["train"]
    n_classes = len(class_names(dataset))
    out_dir = results_dir(dataset)

    configs = [c for c in settings(n_classes, y_train) if not only or c[0] in only]  # --settings lets you run a subset
    max_counts = [max((c[2][k] for c in configs if c[2] is not None), default=0) for k in range(n_classes)]  # largest request per class -> pool size
    pools = {tag: SyntheticPool(load_generator(CKPT_DIR / dataset / f"G_{tag}.pt"), n_classes, max_counts)  # generated ONCE, reused by every ratio and seed
             for tag in ("final", "early")}

    rows, summaries, confusions = [], [], {}
    for name, tag, counts, classical in configs:
        x, y, is_syn = build_training_set(x_train, y_train, pools.get(tag), counts or [0] * n_classes)  # real + synthetic training set
        summaries.append(summarize(name, y, is_syn, n_classes))
        cms = []
        for seed in seeds:  # repeat with different seeds -> mean ± std, so we can tell a real effect from random noise
            t0 = time.time()
            model, history = train_classifier((x, y), data["val"], n_classes, seed, epochs=epochs, classical=classical)
            metrics, cm = evaluate(model, data["test"])  # SAME real test set for every setting
            cms.append(cm.tolist())
            rows.append({"setting": name, "seed": seed, "n_train": len(y), **metrics})
            if name == "real" and seed == seeds[0]:
                pd.DataFrame(history).to_csv(out_dir / "baseline_history.csv", index=False)  # learning curve of the baseline
            print(f"[{dataset}] {name:<18} seed={seed}  acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}  "
                  f"auc={metrics['auc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        confusions[name] = cms

    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "metrics_runs.csv", index=False)  # every single run (setting x seed)
    summary = runs.groupby("setting", sort=False)[list(METRICS)].agg(["mean", "std"])  # mean and std over seeds per setting
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]  # flatten column names, e.g. f1_mean, f1_std
    summary.to_csv(out_dir / "metrics_summary.csv")
    pd.DataFrame(summaries).to_csv(out_dir / "augmentation_summary.csv", index=False)  # dataset sizes / class counts per setting
    (out_dir / "confusion_matrices.json").write_text(json.dumps(confusions))  # per setting, per seed
    print(summary.round(4).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="breastmnist")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])  # used: 5 seeds Breast, 3 seeds Pneumonia (CPU time)
    p.add_argument("--epochs", type=int, default=40)  # used: 40 Breast, 15 Pneumonia (9x more images per epoch)
    p.add_argument("--settings", nargs="+", help="subset of setting names to run (default: all)")
    a = p.parse_args()
    run(a.dataset, a.seeds, a.epochs, a.settings)


if __name__ == "__main__":
    main()
