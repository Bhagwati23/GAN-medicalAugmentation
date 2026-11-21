"""Builds training sets that mix real images with a controlled amount of GAN images (issue #5).

Rules that keep the experiment fair:
  1. Synthetic images are added ONLY to the training split; validation and test stay 100% real.
  2. "X% synthetic" means X% of the real training-set size, e.g. BreastMNIST 546 real -> +50% = 274 (74 + 200, Python rounds 73.5 to even 74), +100% = 546, +200% = 1092 synthetic.
  3. The real class ratio is kept (proportional mode), unless we deliberately balance the classes (balance mode).
  4. One synthetic pool per generator, made with a fixed seed; every ratio takes a PREFIX of it per class, so the
     50% set is inside the 100% set, which is inside the 200% set (nested) -> differences between ratios come from
     the AMOUNT of data, not from drawing a luckier batch of images.
"""
import torch

from .data import class_counts
from .dcgan import generate

RATIOS = (0.0, 0.5, 1.0, 2.0)  # 0%, 50%, 100%, 200% synthetic images (from the project brief)


def synthetic_counts(y_real, n_classes, ratio, mode="proportional"):
    """Number of synthetic images per class.

    proportional: ratio * real count of each class, preserving the real class distribution.
    balance:      only enough minority-class images to equalise the classes (ratio ignored).
    """
    real = class_counts(y_real, n_classes)  # e.g. BreastMNIST [147 malignant, 399 benign]
    if mode == "proportional":
        return [round(ratio * c) for c in real]  # ratio 1.0 -> [147, 399]: same 27/73 split as the real data
    if mode == "balance":
        return [max(real) - c for c in real]  # -> [252, 0]: 252 synthetic malignant so both classes have 399
    raise ValueError(f"unknown mode {mode!r}")


class SyntheticPool:  # pre-generated synthetic images, one tensor per class, generated once and reused by every ratio
    def __init__(self, G, n_classes, max_per_class, seed=1234):
        self.images = {}
        for c in range(n_classes):
            labels = torch.full((max_per_class[c],), c, dtype=torch.long)  # ask G for max_per_class[c] images of class c
            self.images[c] = generate(G, labels, seed=seed + c)  # fixed seed per class -> identical pool on every run

    def take(self, counts):  # first counts[c] images of each class (prefix -> nested subsets)
        xs = [self.images[c][:k] for c, k in enumerate(counts) if k > 0]
        ys = [torch.full((k,), c, dtype=torch.long) for c, k in enumerate(counts) if k > 0]  # label = class the generator was asked for
        if not xs:
            return None, None  # nothing requested
        return torch.cat(xs), torch.cat(ys)


def build_training_set(x_real, y_real, pool, counts):
    """Returns (x, y, is_synthetic) for real training data plus `counts` synthetic images per class."""
    x_syn, y_syn = pool.take(counts) if pool is not None else (None, None)
    if x_syn is None:  # 0% setting -> real data only
        return x_real, y_real, torch.zeros(len(y_real), dtype=torch.bool)
    x = torch.cat([x_real, x_syn])  # real first, then synthetic (DataLoader shuffles them together during training)
    y = torch.cat([y_real, y_syn])
    is_syn = torch.cat([torch.zeros(len(y_real), dtype=torch.bool), torch.ones(len(y_syn), dtype=torch.bool)])  # mask: which samples are fake (for bookkeeping/tests)
    return x, y, is_syn


def summarize(name, y, is_syn, n_classes):  # one row of results/<dataset>/augmentation_summary.csv (dataset sizes per setting)
    real = class_counts(y[~is_syn], n_classes)
    syn = class_counts(y[is_syn], n_classes)
    row = {"setting": name, "n_real": int((~is_syn).sum()), "n_synthetic": int(is_syn.sum()), "n_total": len(y)}
    for c in range(n_classes):
        row[f"class{c}_real"] = real[c]
        row[f"class{c}_synthetic"] = syn[c]
    return row
