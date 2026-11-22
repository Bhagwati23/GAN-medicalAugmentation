"""Central settings shared by every module: folder paths, dataset names, latent size, seeding."""
import random  # Python's own RNG (used e.g. by shuffles inside libraries)
from pathlib import Path  # OS-independent paths (works on Windows, Linux, macOS)

import numpy as np  # NumPy RNG is seeded too so sklearn/numpy code is reproducible
import torch  # PyTorch: tensors, neural networks, training

ROOT = Path(__file__).resolve().parents[1]  # repository root = folder above ganaug/
DATA_DIR = ROOT / "data"  # MedMNIST .npz files are downloaded here (git-ignored, auto-downloaded)
CKPT_DIR = ROOT / "checkpoints"  # trained generator weights G_early.pt / G_final.pt (git-ignored, recreated by training)
RESULTS_DIR = ROOT / "results"  # every metric, table and figure used in the report (committed to git)

DATASETS = ("breastmnist", "pneumoniamnist")  # BreastMNIST = main small dataset (546 train imgs); PneumoniaMNIST = larger 2nd dataset (4708 train imgs)
Z_DIM = 100  # length of the random noise vector z fed to the generator (standard DCGAN value)
IMG_SIZE = 28  # MedMNIST default resolution 28x28 pixels, grayscale (1 channel)


def set_seed(seed):  # same seed -> same weights init, same shuffling -> same results (reproducibility)
    random.seed(seed)  # seed Python RNG
    np.random.seed(seed)  # seed NumPy RNG
    torch.manual_seed(seed)  # seed PyTorch RNG (weight init, dropout, torch.randn noise)


def results_dir(dataset):  # results/<dataset>/ folder, created if missing
    path = RESULTS_DIR / dataset  # e.g. results/breastmnist
    path.mkdir(parents=True, exist_ok=True)  # no error if it already exists
    return path
