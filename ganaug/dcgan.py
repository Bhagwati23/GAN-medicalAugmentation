"""Class-conditional DCGAN (Radford et al. 2016 + Mirza & Osindero 2014 conditioning) for 28x28 grayscale images.

Generator     : z (100) ++ embed(y) (50) -> FC -> 256x7x7 -> ConvT -> 128x14x14 -> ConvT -> 64x28x28 -> Conv -> 1x28x28 (tanh)
Discriminator : x (1x28x28) ++ embed(y) as a 1x28x28 map -> 64x14x14 -> 128x7x7 -> 256x4x4 -> linear -> real/fake logit
"Conditional" = both networks see the class label, so we can ask G for a *malignant* or a *benign* image on demand,
which is what lets us add synthetic images while keeping the real class ratio (or balancing the classes).
"""
import torch
from torch import nn

from .config import IMG_SIZE, Z_DIM


def init_weights(m):  # DCGAN paper rule: all weights ~ Normal(0, 0.02); BatchNorm gamma ~ Normal(1, 0.02)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)  # small random weights -> stable start of adversarial training
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, 1.0, 0.02)  # scale close to 1
        nn.init.zeros_(m.bias)  # shift 0


class Generator(nn.Module):  # noise + label -> fake image   (~2.56 M parameters, most in the first Linear layer)
    def __init__(self, n_classes=2, z_dim=Z_DIM, ch=64, emb_dim=50):
        super().__init__()
        self.z_dim, self.ch = z_dim, ch  # z_dim=100 noise size; ch=64 base number of feature maps
        self.emb = nn.Embedding(n_classes, emb_dim)  # learnable 50-d vector per class (the "condition")
        self.fc = nn.Sequential(  # project noise+label to a small 7x7 feature map
            nn.Linear(z_dim + emb_dim, ch * 4 * 7 * 7, bias=False),  # 150 -> 256*7*7
            nn.BatchNorm1d(ch * 4 * 7 * 7),  # BatchNorm in G (DCGAN guideline) stabilises training
            nn.ReLU(True),  # ReLU in G hidden layers (DCGAN guideline)
        )
        self.net = nn.Sequential(  # upsample 7 -> 14 -> 28
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1, bias=False),  # transposed conv, stride 2: 256x7x7 -> 128x14x14 (learned upsampling)
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1, bias=False),  # 128x14x14 -> 64x28x28
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, 1, 3, 1, 1),  # 64 -> 1 channel (grayscale), size stays 28x28
            nn.Tanh(),  # output in [-1, 1], same range as the real images after scaling
        )
        self.apply(init_weights)  # apply DCGAN init to every layer

    def forward(self, z, y):  # z: (B, 100) noise, y: (B,) class ids -> (B, 1, 28, 28) images
        h = self.fc(torch.cat([z, self.emb(y)], dim=1)).view(-1, self.ch * 4, 7, 7)  # concat noise + label embedding, reshape to 256x7x7
        return self.net(h)


class Discriminator(nn.Module):  # image + label -> one score "how real is this image for this class"   (~0.4 M parameters)
    def __init__(self, n_classes=2, ch=64, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(n_classes, IMG_SIZE * IMG_SIZE)  # label -> 784 values -> reshaped to an extra 28x28 image channel
        self.net = nn.Sequential(
            nn.Conv2d(2, ch, 4, 2, 1),  # 2 input channels (image + label map) -> 64x14x14, stride-2 conv instead of pooling (DCGAN)
            nn.LeakyReLU(0.2, True),  # LeakyReLU in D (DCGAN guideline): gradients flow also for negative inputs; no BN on first layer
            nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),  # -> 128x7x7
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1, bias=False),  # -> 256x4x4
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),  # -> 4096
            nn.Dropout(dropout),  # makes D weaker/noisier so it cannot simply memorise the 546 real images
            nn.Linear(ch * 4 * 4 * 4, 1),  # single logit; sigmoid(logit) = probability "real" (sigmoid is inside BCEWithLogitsLoss)
        )
        self.apply(init_weights)

    def forward(self, x, y):  # x: (B,1,28,28), y: (B,) -> (B,) logits
        label_map = self.emb(y).view(-1, 1, IMG_SIZE, IMG_SIZE)  # label as an image-shaped channel
        return self.net(torch.cat([x, label_map], dim=1)).squeeze(1)  # stack as 2 channels, score it


@torch.no_grad()  # no gradients needed when only sampling -> faster, less memory
def generate(G, labels, seed=0, batch_size=512):
    """Deterministically generates one image per entry of `labels`, in [-1, 1]."""
    G.eval()  # eval mode: BatchNorm uses running statistics, so output does not depend on batch composition
    gen = torch.Generator().manual_seed(seed)  # private RNG -> same seed gives exactly the same synthetic images (reproducible)
    out = []
    for chunk in labels.split(batch_size):  # generate in chunks to limit memory
        z = torch.randn(len(chunk), G.z_dim, generator=gen)  # fresh noise vector per image -> diversity
        out.append(G(z, chunk))
    return torch.cat(out) if out else torch.empty(0, 1, IMG_SIZE, IMG_SIZE)


def load_generator(path):  # checkpoint file -> ready-to-use generator
    ckpt = torch.load(path, map_location="cpu")  # dict saved by train_gan.save_checkpoint
    G = Generator(n_classes=ckpt["n_classes"])
    G.load_state_dict(ckpt["generator"])  # restore trained weights
    return G.eval()
