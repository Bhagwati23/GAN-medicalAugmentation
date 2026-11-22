"""Quality of generated images (issue #6): FID, KID and a memorisation (nearest-neighbour) check.

    python -m ganaug.fid --dataset breastmnist

FID (Heusel et al. 2017): pass real and generated images through Inception-v3 (ImageNet-trained), take the 2048-d
features, fit a Gaussian to each set and measure the Frechet distance between the two Gaussians. Lower = closer to real.
KID (Binkowski et al. 2018): kernel (MMD) distance on the same features; unbiased, so more reliable for small sets like ours.
All sets go through the same preprocessing: [-1, 1] float -> uint8 -> 3 channels; torch-fidelity resizes to 299x299.
Real-train vs real-test is the REFERENCE FLOOR: even two sets of real images get FID ~59 here, because 28x28 grayscale
ultrasound is far from ImageNet photos -> compare our FIDs with this floor, not with natural-image papers.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch_fidelity  # standard, tested FID/KID implementation (same numbers as the original TensorFlow code)

from .config import CKPT_DIR, results_dir
from .data import class_counts, class_names, load_split
from .dcgan import generate, load_generator


class UInt8Images(torch.utils.data.Dataset):  # torch-fidelity expects uint8 tensors of shape (3, H, W)
    def __init__(self, x):
        self.x = x.add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8).repeat(1, 3, 1, 1)  # [-1,1] -> 0..255, grayscale copied to 3 RGB channels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


def fid_kid(x_a, x_b):
    subset = min(len(x_a), len(x_b), 1000) // 2  # KID is averaged over random subsets of this size
    m = torch_fidelity.calculate_metrics(
        input1=UInt8Images(x_a), input2=UInt8Images(x_b), fid=True, kid=True,
        kid_subset_size=subset, kid_subsets=50, cuda=False, verbose=False, cache=False,  # 50 random subsets -> KID mean ± std
    )
    return {"fid": m["frechet_inception_distance"], "kid_mean": m["kernel_inception_distance_mean"],
            "kid_std": m["kernel_inception_distance_std"]}


def nearest_distances(queries, refs):  # for every query image: pixel L2 distance to its closest reference image
    q, r = queries.flatten(1), refs.flatten(1)  # (N, 784) vectors
    return torch.cdist(q, r).min(dim=1)  # (values, indices of the nearest reference)


def memorisation(x_train, x_test, x_fake):
    """Does G copy training images? Compare fake->train nearest distances with test->train distances.
    Test images are real but unseen, so their distance to the training set is what 'new, realistic' looks like.
    If fakes were much closer to training images than that (e.g. below the test 5th percentile), G would be memorising."""
    d_fake = nearest_distances(x_fake, x_train).values
    d_test = nearest_distances(x_test, x_train).values
    threshold = torch.quantile(d_test, 0.05)  # only 5% of real unseen images are this close to a training image
    return {"median_nn_dist_fake_to_train": d_fake.median().item(),
            "median_nn_dist_test_to_train": d_test.median().item(),
            "frac_fake_closer_than_test_p5": (d_fake < threshold).float().mean().item()}  # ~0.05 or less = no sign of copying


def plot_nearest_neighbours(x_train, x_fake, out, n=8):  # top row: generated, bottom row: its closest real training image
    idx = nearest_distances(x_fake[:n], x_train).indices
    fig, axes = plt.subplots(2, n, figsize=(n * 1.1, 2.6))
    for i in range(n):
        for row, img in enumerate((x_fake[i], x_train[idx[i]])):
            axes[row, i].imshow(img[0], cmap="gray", vmin=-1, vmax=1)
            axes[row, i].axis("off")
    axes[0, 0].set_title("generated", fontsize=8, loc="left")
    axes[1, 0].set_title("nearest real (train)", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def evaluate_dataset(dataset, seed=2024, max_n=2000):
    x_train, y_train = load_split(dataset, "train")
    if len(y_train) > max_n:  # PneumoniaMNIST: random fixed subset of 2000 real images keeps Inception on CPU fast; BreastMNIST (546) uses all
        keep = torch.randperm(len(y_train), generator=torch.Generator().manual_seed(seed))[:max_n]
        x_train, y_train = x_train[keep], y_train[keep]
    x_test, _ = load_split(dataset, "test")  # used ONLY as the real-vs-real reference, never for training
    n_classes = len(class_names(dataset))
    out_dir = results_dir(dataset)
    results = {"n_real_train": len(x_train), "n_generated": len(x_train), "real_train_vs_real_test": fid_kid(x_train, x_test)}

    labels = torch.cat([torch.full((c,), k, dtype=torch.long) for k, c in enumerate(class_counts(y_train, n_classes))])  # same size and class ratio as the real train set
    for tag in ("early", "final"):
        path = CKPT_DIR / dataset / f"G_{tag}.pt"
        if not path.exists():
            continue
        G = load_generator(path)
        epoch = torch.load(path, map_location="cpu")["epoch"]
        x_fake = generate(G, labels, seed=seed)
        results[f"generator_{tag}"] = {"epoch": epoch, **fid_kid(x_train, x_fake), **memorisation(x_train, x_test, x_fake)}
        if tag == "final":
            plot_nearest_neighbours(x_train, x_fake, out_dir / "fig_nearest_neighbours.png")

    (out_dir / "fid.json").write_text(json.dumps(results, indent=2))  # results/<dataset>/fid.json
    print(json.dumps(results, indent=2))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="breastmnist")
    evaluate_dataset(p.parse_args().dataset)


if __name__ == "__main__":
    main()
