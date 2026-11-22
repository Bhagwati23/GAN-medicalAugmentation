"""Train the conditional DCGAN on the real *training* split only (val/test are never shown to the GAN -> no data leakage).

    python -m ganaug.train_gan --dataset breastmnist --epochs 300 --early-epoch 30

Saves two generators: G_early (after ~10% of training, worse images, higher FID) and G_final (end of training,
better images, lower FID). Using both lets the experiments separate the effect of synthetic-data QUALITY from QUANTITY.
"""
import argparse  # command-line options
import csv  # write per-epoch losses
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image  # saves a tensor batch as a PNG grid

from .config import CKPT_DIR, Z_DIM, results_dir, set_seed
from .data import as_dataset, class_names, load_split, translate
from .dcgan import Discriminator, Generator


def save_checkpoint(G, n_classes, epoch, path):  # only G is saved: D is not needed after training
    torch.save({"generator": G.state_dict(), "n_classes": n_classes, "epoch": epoch}, path)


def train(dataset, epochs, batch_size=64, lr=2e-4, seed=0, early_epoch=None, sample_every=25,
          real_label=0.9, diffaug=True):
    set_seed(seed)  # reproducible GAN training
    x, y = load_split(dataset, "train")  # ONLY the training split
    n_classes = len(class_names(dataset))
    loader = DataLoader(as_dataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True)  # drop_last: avoid a tiny last batch (bad for BatchNorm)

    G, D = Generator(n_classes), Discriminator(n_classes)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))  # DCGAN settings: lr 2e-4, beta1 0.5 (0.9 makes GANs oscillate)
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = torch.nn.BCEWithLogitsLoss()  # binary cross-entropy on logits (sigmoid inside, numerically stable)
    aug = translate if diffaug else (lambda t: t)  # DiffAugment (Zhao et al. 2020): shift BOTH real and fake before D, so D can't memorise 546 images

    ckpt_dir = CKPT_DIR / dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = results_dir(dataset)
    sample_dir = out_dir / "gan_samples"  # PNG grid every `sample_every` epochs: shows training progress visually
    sample_dir.mkdir(exist_ok=True)

    per_class = 8  # 8 images per class in each sample grid (row 1 = class 0, row 2 = class 1)
    fixed_y = torch.arange(n_classes).repeat_interleave(per_class)  # [0]*8 + [1]*8
    fixed_z = torch.randn(len(fixed_y), Z_DIM, generator=torch.Generator().manual_seed(seed))  # SAME noise every time -> grids are comparable across epochs
    early_epoch = early_epoch or max(1, epochs // 10)  # default early checkpoint at 10% of training

    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        d_sum = g_sum = dx_sum = dgz_sum = 0.0  # running sums for epoch averages
        for xb, yb in loader:
            n = xb.size(0)
            z = torch.randn(n, Z_DIM)  # random noise
            fake = G(z, yb)  # fake images with the same labels as the real batch (keeps class ratio)

            # ---- Step 1: update D -> say "real" (0.9) for real images, "fake" (0) for generated ones
            d_real = D(aug(xb), yb)
            d_fake = D(aug(fake.detach()), yb)  # detach: this step must not update G
            loss_d = bce(d_real, torch.full((n,), real_label)) + bce(d_fake, torch.zeros(n))  # one-sided label smoothing: target 0.9 not 1.0 -> D less over-confident
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # ---- Step 2: update G -> fool D into saying "real" (target 1) for fake images
            loss_g = bce(D(aug(fake), yb), torch.ones(n))  # non-saturating GAN loss (Goodfellow et al. 2014)
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            d_sum += loss_d.item(); g_sum += loss_g.item()
            dx_sum += torch.sigmoid(d_real).mean().item(); dgz_sum += torch.sigmoid(d_fake).mean().item()  # D(x): D's avg prob for real; D(G(z)): for fake

        steps = len(loader)
        row = {"epoch": epoch, "loss_d": d_sum / steps, "loss_g": g_sum / steps,  # ideal equilibrium: D(x) ~ D(G(z)) ~ 0.5 (D can't tell them apart)
               "d_real": dx_sum / steps, "d_fake": dgz_sum / steps}
        history.append(row)

        if epoch == 1 or epoch % sample_every == 0 or epoch in (early_epoch, epochs):
            G.eval()
            with torch.no_grad():
                imgs = G(fixed_z, fixed_y)
            save_image(imgs, sample_dir / f"epoch_{epoch:04d}.png", nrow=per_class, normalize=True, value_range=(-1, 1))  # map [-1,1] -> [0,1] for PNG
            print(f"[{dataset}] epoch {epoch}/{epochs}  D={row['loss_d']:.3f}  G={row['loss_g']:.3f}  "
                  f"D(x)={row['d_real']:.2f}  D(G(z))={row['d_fake']:.2f}  {time.time() - start:.0f}s", flush=True)
        if epoch == early_epoch:
            save_checkpoint(G, n_classes, epoch, ckpt_dir / "G_early.pt")  # lower-quality generator for the QUALITY axis

    save_checkpoint(G, n_classes, epochs, ckpt_dir / "G_final.pt")  # fully trained generator
    with open(out_dir / "gan_losses.csv", "w", newline="") as f:  # per-epoch losses -> results/<dataset>/gan_losses.csv (plotted in fig_gan_losses.png)
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    print(f"[{dataset}] done in {time.time() - start:.0f}s; early checkpoint at epoch {early_epoch}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="breastmnist")
    p.add_argument("--epochs", type=int, default=300)  # used: 300 for BreastMNIST (546 imgs), 60 for PneumoniaMNIST (4708 imgs) -> similar number of updates
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--early-epoch", type=int, default=None)  # used: 30 (Breast), 6 (Pneumonia) = 10% of training
    p.add_argument("--sample-every", type=int, default=25)
    p.add_argument("--no-diffaug", action="store_true")  # switch DiffAugment off (for ablation)
    a = p.parse_args()
    train(a.dataset, a.epochs, a.batch_size, a.lr, a.seed, a.early_epoch, a.sample_every, diffaug=not a.no_diffaug)


if __name__ == "__main__":
    main()
