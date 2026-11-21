"""Fast checks (~10 s, no dataset download) that the pieces fit together and the experiment rules hold.  Run: pytest -q"""
import torch

from ganaug.augment import SyntheticPool, build_training_set, synthetic_counts
from ganaug.cnn import CNNClassifier
from ganaug.dcgan import Discriminator, Generator, generate


def test_generator_and_discriminator_shapes():  # issue #3 acceptance: forward passes work without shape errors
    G, D = Generator(n_classes=2), Discriminator(n_classes=2)
    z, y = torch.randn(4, G.z_dim), torch.tensor([0, 1, 0, 1])  # 4 noise vectors + their class labels
    fake = G(z, y)
    assert fake.shape == (4, 1, 28, 28)  # same shape as a real MedMNIST image
    assert fake.min() >= -1 and fake.max() <= 1  # tanh range = real image range
    assert D(fake, y).shape == (4,)  # one real/fake score per image


def test_classifier_output_shape():  # one logit per class
    assert CNNClassifier(n_classes=3)(torch.randn(5, 1, 28, 28)).shape == (5, 3)


def test_generation_is_deterministic():  # same seed -> identical synthetic images (reproducibility)
    G = Generator(n_classes=2)
    labels = torch.tensor([0, 1, 1])
    assert torch.equal(generate(G, labels, seed=3), generate(G, labels, seed=3))


def test_synthetic_counts_preserve_class_ratio_and_balance():  # issue #5: class distribution is preserved / balanced
    y = torch.tensor([0] * 10 + [1] * 30)  # toy imbalanced set: 10 vs 30
    assert synthetic_counts(y, 2, 0.5) == [5, 15]  # +50% keeps the 1:3 ratio
    assert synthetic_counts(y, 2, 2.0) == [20, 60]  # +200%
    assert synthetic_counts(y, 2, 0, mode="balance") == [20, 0]  # only minority images, until 30 vs 30


def test_augmented_sets_are_nested_and_only_touch_training_data():  # issue #5/#8: no leakage, nested subsets
    G = Generator(n_classes=2)
    x_real = torch.rand(40, 1, 28, 28) * 2 - 1
    y_real = torch.tensor([0] * 10 + [1] * 30)
    pool = SyntheticPool(G, 2, max_per_class=[20, 60])

    x50, y50, s50 = build_training_set(x_real, y_real, pool, synthetic_counts(y_real, 2, 0.5))
    x200, y200, s200 = build_training_set(x_real, y_real, pool, synthetic_counts(y_real, 2, 2.0))

    assert len(y50) == 60 and int(s50.sum()) == 20  # 40 real + 20 synthetic
    assert len(y200) == 120 and int(s200.sum()) == 80  # 40 real + 80 synthetic
    assert torch.equal(x50[:40], x_real) and torch.equal(y50[:40], y_real)  # real images are kept unchanged
    syn50_class0 = x50[s50 & (y50 == 0)]
    syn200_class0 = x200[s200 & (y200 == 0)]
    assert torch.equal(syn50_class0, syn200_class0[: len(syn50_class0)])  # the 50% synthetic images are the first part of the 200% set

    x0, y0, s0 = build_training_set(x_real, y_real, pool, [0, 0])  # 0% setting = untouched real data
    assert torch.equal(x0, x_real) and not s0.any()
