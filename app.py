"""Interactive demo (Gradio web app) for the project: generate synthetic scans, play "Spot the Fake",
classify images with two models side by side, browse results, and see how the pipeline works.

    python app.py            -> opens http://127.0.0.1:7860 in the browser

Needs the trained generators (python -m ganaug.train_gan ...). The two demo classifiers (real-only and
real + 100% synthetic, seed 0) are trained once on first use (~1-3 min on CPU) and cached in checkpoints/.
Everything runs locally on the laptop; no image leaves the machine.
"""
import json

import gradio as gr  # web UI (frontend) + local server (backend) in one Python file
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from ganaug.augment import SyntheticPool, build_training_set, synthetic_counts
from ganaug.classifier import predict, train_classifier
from ganaug.cnn import CNNClassifier
from ganaug.config import CKPT_DIR, DATASETS, RESULTS_DIR, Z_DIM
from ganaug.data import class_names, load_dataset
from ganaug.dcgan import generate, load_generator

MODELS = {"real": "Model A · trained on real images only", "gan_final_100": "Model B · real + 100% GAN images"}
PRETTY = {"breastmnist": "BreastMNIST · breast ultrasound", "pneumoniamnist": "PneumoniaMNIST · chest X-ray"}
_cache = {}  # loaded generators / classifiers / datasets, so every click does not reload them from disk


def cached(key, make):  # tiny memo helper: build an object once, reuse it afterwards
    if key not in _cache:
        _cache[key] = make()
    return _cache[key]


def generator(dataset, quality):  # early = epoch-10% checkpoint (worse), final = end of training (better)
    return cached(("G", dataset, quality), lambda: load_generator(CKPT_DIR / dataset / f"G_{quality}.pt"))


def data(dataset):
    return cached(("data", dataset), lambda: load_dataset(dataset))


def to_image(t, scale=4):  # tensor in [-1, 1] -> upscaled uint8 numpy image (nearest = crisp pixels)
    if t.dim() == 3:
        t = t[None]
    t = F.interpolate(t, scale_factor=scale, mode="nearest")[0]
    return t.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).squeeze(-1).numpy()


# ----------------------------------------------------------------------------------------------- Generate tab
def generate_images(dataset, class_name, quality, n, seed):
    names = class_names(dataset)
    labels = torch.full((int(n),), names.index(class_name), dtype=torch.long)  # conditional GAN: request this class
    imgs = generate(generator(dataset, quality), labels, seed=int(seed))
    return [(to_image(img), f"{class_name} · #{i + 1}") for i, img in enumerate(imgs)]  # gallery items (image, caption)


@torch.no_grad()
def latent_walk(dataset, class_name, seed, steps=8):  # straight line z0 -> z1 through the latent space
    G = generator(dataset, "final")
    gen = torch.Generator().manual_seed(int(seed))
    z0, z1 = torch.randn(2, Z_DIM, generator=gen)
    t = torch.linspace(0, 1, steps).unsqueeze(1)
    y = torch.full((steps,), class_names(dataset).index(class_name))
    strip = make_grid(G((1 - t) * z0 + t * z1, y), nrow=steps, pad_value=1)  # smooth change = no memorisation
    return to_image(strip, scale=3)


# ----------------------------------------------------------------------------------------- Spot-the-fake game
def new_round(dataset, state):  # one real training image vs one GAN image of the same class, random sides
    rng = np.random.default_rng()
    x, y = data(dataset)["train"]
    i = int(rng.integers(len(y)))
    fake = generate(generator(dataset, "final"), y[i:i + 1], seed=int(rng.integers(1_000_000)))[0]
    real_left = bool(rng.integers(2))
    left, right = (x[i], fake) if real_left else (fake, x[i])
    state = {**state, "real_left": real_left, "answered": False}
    label = class_names(dataset)[y[i]]
    return to_image(left, 6), to_image(right, 6), f"Both images are labelled **{label}**. Which one is the real scan?", state


def answer(choice_left, state):
    if state.get("answered", True):
        return "Press **New round** first.", state
    correct = choice_left == state["real_left"]
    state = {**state, "answered": True, "rounds": state["rounds"] + 1, "score": state["score"] + int(correct)}
    verdict = "✅ Correct — you spotted the real scan!" if correct else "❌ Fooled! That one was made by the GAN."
    rate = 100 * state["score"] / state["rounds"]
    return f"{verdict}\n\n**Score: {state['score']} / {state['rounds']}** ({rate:.0f}%). " \
           f"Around 50% means the GAN fools you as often as a coin flip.", state


# -------------------------------------------------------------------------------------------- Classify tab
def classifier_for(dataset, setting):  # train once (seed 0, same recipe as ganaug.experiments), then cache
    def make():
        path = CKPT_DIR / dataset / f"clf_{setting}.pt"
        n_classes = len(class_names(dataset))
        model = CNNClassifier(n_classes)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location="cpu"))
            return model.eval()
        d = data(dataset)
        x, y = d["train"]
        counts = synthetic_counts(y, n_classes, 1.0) if setting == "gan_final_100" else [0] * n_classes
        pool = SyntheticPool(generator(dataset, "final"), n_classes, counts) if any(counts) else None
        x, y, _ = build_training_set(x, y, pool, counts)
        model, _ = train_classifier((x, y), d["val"], n_classes, seed=0, epochs=40 if dataset == "breastmnist" else 15)
        torch.save(model.state_dict(), path)
        return model.eval()
    return cached(("clf", dataset, setting), make)


def random_test_image(dataset):  # real TEST image: never used for training the GAN or the classifiers
    x, y = data(dataset)["test"]
    i = int(np.random.default_rng().integers(len(y)))
    return to_image(x[i], 6), f"Ground truth: **{class_names(dataset)[y[i]]}** (test image #{i})"


def classify(dataset, image, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Pick a random test image or upload one first.")
    progress(0.1, desc="Loading models (first run trains them, ~1-3 min)…")
    img = torch.from_numpy(np.asarray(image, dtype=np.float32))
    if img.ndim == 3:
        img = img.mean(-1)  # colour upload -> grayscale
    img = F.interpolate(img[None, None], size=(28, 28), mode="area")[0] / 127.5 - 1  # any size -> 28x28 in [-1, 1]
    names = class_names(dataset)
    return [{names[k]: float(p) for k, p in enumerate(predict(classifier_for(dataset, s), img[None])[0])} for s in MODELS]


# --------------------------------------------------------------------------------------------- Results tab
def load_results(dataset):
    d = RESULTS_DIR / dataset
    summary = pd.read_csv(d / "metrics_summary.csv", index_col=0) if (d / "metrics_summary.csv").exists() else None
    fid = json.loads((d / "fid.json").read_text()) if (d / "fid.json").exists() else None
    return summary, fid


def stat_cards(dataset):  # headline numbers as coloured cards (HTML)
    summary, fid = load_results(dataset)
    if summary is None:
        return "<div class='card'>Run the experiments to see results.</div>"
    best = summary["f1_mean"].idxmax()
    cards = [
        ("Real-only macro F1", f"{summary.loc['real', 'f1_mean']:.3f}", "baseline CNN, real test set"),
        ("Best macro F1", f"{summary.loc[best, 'f1_mean']:.3f}", best.replace("_", " ")),
        ("Best ROC-AUC", f"{summary['auc_mean'].max():.3f}", summary["auc_mean"].idxmax().replace("_", " ")),
    ]
    if fid:
        cards.append(("FID final generator", f"{fid['generator_final']['fid']:.0f}",
                      f"real-vs-real floor {fid['real_train_vs_real_test']['fid']:.0f} · early {fid['generator_early']['fid']:.0f}"))
    return "<div class='cards'>" + "".join(
        f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div><div class='s'>{s}</div></div>" for k, v, s in cards) + "</div>"


def results_view(dataset):
    d = RESULTS_DIR / dataset
    table = (d / "results_table.md").read_text(encoding="utf-8") if (d / "results_table.md").exists() else "_No results yet._"
    figs = [str(d / f) for f in ("fig_metrics_vs_ratio.png", "fig_confusion.png", "fig_real_vs_synthetic.png",
                                 "fig_gan_losses.png", "fig_nearest_neighbours.png", "fig_interpolation.png") if (d / f).exists()]
    return stat_cards(dataset), table, figs


# --------------------------------------------------------------------------------------------------- Layout
CSS = """
.hero {padding: 28px 32px; border-radius: 18px; color: #fff;
       background: linear-gradient(120deg, #312e81 0%, #6d28d9 45%, #db2777 100%);
       box-shadow: 0 10px 30px rgba(109, 40, 217, .25);}
.hero h1 {font-size: 2.1rem; margin: 0 0 6px 0; color: #fff;}
.hero p {opacity: .92; margin: 0; font-size: 1.02rem; color: #fff;}
.badges {margin-top: 14px;}
.badge {display: inline-block; padding: 4px 12px; margin: 4px 6px 0 0; border-radius: 999px;
        background: rgba(255, 255, 255, .18); font-size: .82rem; color: #fff;}
.cards {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 6px 0;}
.card {border-radius: 14px; padding: 16px 18px; background: var(--block-background-fill);
       border: 1px solid var(--border-color-primary); box-shadow: 0 2px 10px rgba(0, 0, 0, .05);}
.card .k {font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; opacity: .7;}
.card .v {font-size: 2rem; font-weight: 700; background: linear-gradient(90deg, #6d28d9, #db2777);
          -webkit-background-clip: text; background-clip: text; color: transparent;}
.card .s {font-size: .85rem; opacity: .75;}
.steps {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px;}
.step {border-radius: 14px; padding: 14px; border: 1px solid var(--border-color-primary); background: var(--block-background-fill);}
.step b {display: block; margin-bottom: 4px;}
.step .n {display: inline-block; width: 26px; height: 26px; border-radius: 50%; text-align: center; line-height: 26px;
          color: #fff; background: linear-gradient(120deg, #6d28d9, #db2777); margin-bottom: 6px; font-weight: 700;}
.footer {text-align: center; opacity: .6; font-size: .85rem;}
"""

HERO = """
<div class='hero'>
  <h1>🧬 Does Synthetic Data Help?</h1>
  <p>Class-conditional DCGAN augmentation for small medical image datasets — generate scans, try to spot the fake,
     and compare a classifier trained with and without GAN images.</p>
  <div class='badges'>
    <span class='badge'>MedMNIST v2</span><span class='badge'>Conditional DCGAN</span>
    <span class='badge'>FID · KID · memorisation check</span><span class='badge'>9 settings × 5 seeds</span>
    <span class='badge'>Team 33 · P33</span>
  </div>
</div>
"""

HOW = """
<div class='steps'>
  <div class='step'><span class='n'>1</span><b>Real data</b>MedMNIST official train / val / test splits, 28×28 grayscale. Test images are never used for training.</div>
  <div class='step'><span class='n'>2</span><b>Conditional DCGAN</b>Learns to draw images of a requested class from random noise; trained on the training split only.</div>
  <div class='step'><span class='n'>3</span><b>Quality check</b>FID / KID vs a real-vs-real floor, and a nearest-neighbour test that the GAN does not copy training images.</div>
  <div class='step'><span class='n'>4</span><b>Augment</b>Add 0 / 50 / 100 / 200 % synthetic images (class ratio kept), or balance the classes.</div>
  <div class='step'><span class='n'>5</span><b>Classify</b>Same CNN for every setting, best epoch by validation AUC, several seeds.</div>
  <div class='step'><span class='n'>6</span><b>Compare</b>Accuracy, macro F1, AUC, confusion matrices and Welch t-tests on the same real test set.</div>
</div>
"""

with gr.Blocks(title="GAN Augmentation · Team 33") as demo:
    gr.HTML(HERO)
    with gr.Row():
        dataset = gr.Dropdown([(PRETTY[d], d) for d in DATASETS], value=DATASETS[0], label="Dataset", scale=2)
    cards = gr.HTML(stat_cards(DATASETS[0]))

    with gr.Tab("🧬 Generate"):
        with gr.Row():
            cls = gr.Dropdown(class_names(DATASETS[0]), value=class_names(DATASETS[0])[0], label="Class")
            quality = gr.Radio([("Final generator (better)", "final"), ("Early generator (10% trained)", "early")],
                               value="final", label="Generator checkpoint")
            n = gr.Slider(4, 32, value=16, step=4, label="How many images")
            seed = gr.Number(value=7, label="Seed", precision=0)
        with gr.Row():
            gen_btn = gr.Button("✨ Generate", variant="primary")
            walk_btn = gr.Button("🌀 Latent walk (z₀ → z₁)")
        gallery = gr.Gallery(label="Synthetic images", columns=8, height="auto")
        walk = gr.Image(label="Latent walk: smooth change means the GAN learned a continuous distribution", show_label=True)
        gen_btn.click(generate_images, [dataset, cls, quality, n, seed], gallery)
        walk_btn.click(latent_walk, [dataset, cls, seed], walk)

    with gr.Tab("🎯 Spot the Fake"):
        gr.Markdown("One image is a **real scan** from the training set, the other was **drawn by the GAN**. Can you tell which is real?")
        state = gr.State({"score": 0, "rounds": 0, "answered": True})
        prompt = gr.Markdown()
        with gr.Row():
            left = gr.Image(label="Left", height=260)
            right = gr.Image(label="Right", height=260)
        with gr.Row():
            pick_left = gr.Button("⬅️ Left is real")
            pick_right = gr.Button("Right is real ➡️")
            round_btn = gr.Button("🔄 New round", variant="primary")
        verdict = gr.Markdown()
        round_btn.click(new_round, [dataset, state], [left, right, prompt, state])
        pick_left.click(lambda s: answer(True, s), state, [verdict, state])
        pick_right.click(lambda s: answer(False, s), state, [verdict, state])

    with gr.Tab("🔬 Classify"):
        gr.Markdown("The same real **test** image goes through two CNNs that differ **only in their training data**.")
        with gr.Row():
            with gr.Column():
                img = gr.Image(label="Image (random real test image, or upload your own)", image_mode="L", height=260)
                pick_btn = gr.Button("🎲 Random test image")
                truth = gr.Markdown()
            with gr.Column():
                out_a = gr.Label(label=MODELS["real"], num_top_classes=2)
                out_b = gr.Label(label=MODELS["gan_final_100"], num_top_classes=2)
                clf_btn = gr.Button("🔬 Classify with both models", variant="primary")
        pick_btn.click(random_test_image, dataset, [img, truth])
        clf_btn.click(classify, [dataset, img], [out_a, out_b])

    with gr.Tab("📊 Results"):
        table = gr.Markdown()
        figs = gr.Gallery(label="Figures from results/", columns=2, height="auto")
        gr.Button("📊 Load results", variant="primary").click(results_view, dataset, [cards, table, figs])

    with gr.Tab("🧭 How it works"):
        gr.HTML(HOW)
        gr.Markdown("Code: `ganaug/` (one module per step) · full reproduction: `bash run_all.sh` · tests: `pytest -q`")

    dataset.change(lambda d: (gr.Dropdown(choices=class_names(d), value=class_names(d)[0]), stat_cards(d)),
                   dataset, [cls, cards])  # switching dataset updates class list and headline cards
    gr.HTML("<div class='footer'>Research / education demo · not a medical device · Team 33, Sitare University</div>")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="violet", secondary_hue="pink",
                                     font=[gr.themes.GoogleFont("Inter"), "sans-serif"]), css=CSS)
