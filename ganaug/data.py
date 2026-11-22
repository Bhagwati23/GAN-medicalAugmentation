"""Dataset loading (MedMNIST v2) and the small image-shift helper shared by the GAN and the classifier.

Splits are the OFFICIAL MedMNIST train/val/test splits, fixed by the dataset authors, so every member and
every run uses exactly the same images. The test split is only ever used for the final evaluation.
"""
import torch
import torch.nn.functional as F  # F.pad used for the random shift
from medmnist import INFO  # metadata per dataset: class names, python class name, task type
from torch.utils.data import TensorDataset  # wraps (images, labels) tensors into a Dataset for DataLoader

from src.data.dataset import load_breast_mnist, load_pneumonia_mnist  # Issue #1 loaders (official splits, uint8 images)

from .config import DATA_DIR

LOADERS = {"breastmnist": load_breast_mnist, "pneumoniamnist": load_pneumonia_mnist}  # dataset name -> Issue #1 loader
SPLITS = ("train", "val", "test")  # train: fit models (and the GAN) | val: pick best epoch | test: final, untouched evaluation


def class_names(name):  # e.g. breastmnist -> ["malignant", "normal, benign"]; label 0 = malignant, 1 = normal/benign
    return [INFO[name]["label"][str(i)] for i in range(len(INFO[name]["label"]))]


def load_split(name, split):
    """Returns images scaled to [-1, 1] with shape (N, 1, 28, 28) and integer labels of shape (N,)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)  # make data/ if missing
    imgs, labels = LOADERS[name](split=split, root=str(DATA_DIR))  # 28x28 uint8 (N,28,28,1) + int64 (N,); downloads once (~0.5 MB Breast, ~4 MB Pneumonia)
    x = torch.from_numpy(imgs).float().div(127.5).sub(1).permute(0, 3, 1, 2)  # uint8 0..255 -> float -1..1 (matches GAN tanh output); channels-last -> (N,1,28,28)
    y = torch.from_numpy(labels).long()  # int64 labels for CrossEntropy/Embedding
    return x, y


def load_dataset(name):
    """All three official splits as {"train": (x, y), "val": (x, y), "test": (x, y)}."""
    return {split: load_split(name, split) for split in SPLITS}


def as_dataset(x, y):  # tensors -> torch Dataset (needed by DataLoader for batching/shuffling)
    return TensorDataset(x, y)


def class_counts(y, n_classes):  # number of images per class, e.g. BreastMNIST train -> [147, 399] (27% malignant = imbalanced)
    return torch.bincount(y, minlength=n_classes).tolist()


def translate(x, max_shift=2):
    """Random per-sample shift of up to `max_shift` pixels (GAN DiffAugment and classifier augmentation)."""
    n = x.size(0)  # batch size
    dx, dy = torch.randint(-max_shift, max_shift + 1, (2, n))  # independent random x/y shift in [-2, 2] for every image
    padded = F.pad(x, [max_shift] * 4, mode="replicate")  # pad 2 px on each side by repeating edge pixels (no black borders)
    h, w = x.shape[-2:]  # 28, 28
    return torch.stack([  # crop a 28x28 window at the shifted position = image moved by (dx, dy)
        padded[i, :, max_shift + dy[i]: max_shift + dy[i] + h, max_shift + dx[i]: max_shift + dx[i] + w]
        for i in range(n)
    ])
