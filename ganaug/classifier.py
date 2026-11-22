"""Classifier training and evaluation shared by the baseline (issue #2) and all augmentation experiments (issue #7).

Model selection: after every epoch we score the VALIDATION set and keep the weights with the best validation AUC
(early-stopping style). The TEST set is used exactly once, at the very end -> no test-set tuning, honest numbers.
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .cnn import CNNClassifier
from .config import set_seed
from .data import translate


def classical_augment(x):
    """Random horizontal flip plus a small random translation, per sample (the non-GAN augmentation baseline)."""
    flip = torch.rand(x.size(0)) < 0.5  # each image flipped left-right with probability 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)  # flip(-1) mirrors the width axis
    return translate(x, max_shift=2)  # plus a random shift of up to 2 px


@torch.no_grad()  # inference only
def predict(model, x, batch_size=1024):  # -> class probabilities, numpy array (N, n_classes)
    model.eval()  # eval mode: dropout off, BatchNorm uses running stats
    probs = [torch.softmax(model(xb), dim=1) for xb in x.split(batch_size)]  # softmax turns logits into probabilities summing to 1
    return torch.cat(probs).numpy()


def compute_metrics(y_true, probs):
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(1)  # predicted class = highest probability (threshold 0.5 for 2 classes)
    if probs.shape[1] == 2:
        auc = roc_auc_score(y_true, probs[:, 1])  # ROC-AUC: threshold-free ranking quality, robust to class imbalance
    else:
        auc = roc_auc_score(y_true, probs, multi_class="ovr")  # one-vs-rest for >2 classes
    return {
        "accuracy": accuracy_score(y_true, y_pred),  # fraction correct (can look good just by predicting the majority class)
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),  # macro = average over classes, each class counts equally
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),  # macro recall = balanced accuracy; drops if the minority (malignant) class is missed
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),  # harmonic mean of precision & recall per class, averaged -> our main metric
        "auc": auc,
    }


def train_classifier(train, val, n_classes, seed, epochs=40, batch_size=64, lr=1e-3, weight_decay=1e-4,
                     classical=False):
    """Trains on `train`, keeps the weights with the best validation AUC. Returns (model, history)."""
    set_seed(seed)  # seed controls weight init + batch order; we repeat every setting with several seeds and report mean ± std
    model = CNNClassifier(n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # AdamW: Adam with decoupled weight decay (L2 regularisation)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)  # learning rate decays smoothly from 1e-3 to ~0 over training
    loss_fn = torch.nn.CrossEntropyLoss()  # standard multi-class loss (softmax + negative log-likelihood); no class weights -> same for every setting
    loader = DataLoader(TensorDataset(*train), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))  # shuffled mini-batches mix real and synthetic images

    best_auc, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()  # dropout + BatchNorm in training mode
        total = 0.0
        for xb, yb in loader:
            if classical:
                xb = classical_augment(xb)  # only for the "classical" setting (fresh random flip/shift every epoch)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()  # standard backprop step
            total += loss.item() * len(yb)
        sched.step()  # update learning rate once per epoch
        val_metrics = compute_metrics(val[1].numpy(), predict(model, val[0]))  # validation = REAL images only
        history.append({"epoch": epoch, "train_loss": total / len(train[1]), **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["auc"] > best_auc:  # new best on validation -> remember these weights
            best_auc = val_metrics["auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # clone = copy, otherwise later training would overwrite it

    model.load_state_dict(best_state)  # return the best-validation model, not the last-epoch one
    return model, history


def evaluate(model, test):  # final, one-time evaluation on the held-out REAL test set
    probs = predict(model, test[0])
    metrics = compute_metrics(test[1].numpy(), probs)
    cm = confusion_matrix(test[1].numpy(), probs.argmax(1))  # rows = true class, columns = predicted class
    return metrics, cm
