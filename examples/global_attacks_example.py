"""Detecting adversarial attacks with subset scanning (ART).

Follows https://opendatascience.com/detecting-adversarial-attacks-with-subset-scanning/:
1. Train a small CNN on MNIST.
2. Craft adversarial samples with FGSM.
3. Learn a null model from background (clean) activations and score each
   clean and adversarial sample individually. Adversarial inputs tend to get
   higher subset-scanning scores.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from art.attacks.evasion import FastGradientMethod
from art.defences.detector.evasion import SubsetScanningDetector
from art.estimators.classification import PyTorchClassifier
from torch import nn
from torchvision.datasets import MNIST

SEED = 0
EPOCHS = 3
EPS_VALUES = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.01]  # decreasing attack strength
FRAME_DURATION_MS = 1000
N_BACKGROUND = 1000
N_SCAN = 100  # samples in each of the clean / adversarial scan sets
N_EXAMPLES = 8  # digits shown in figure

torch.manual_seed(SEED)
np.random.seed(SEED)

# Data: (N, 1, 28, 28) float32 in [0, 1]
train = MNIST(root="data", train=True, download=True)
test = MNIST(root="data", train=False, download=True)
x_train = (train.data.numpy().astype(np.float32) / 255.0)[:, None]
y_train = train.targets.numpy()
x_test = (test.data.numpy().astype(np.float32) / 255.0)[:, None]
y_test = test.targets.numpy()

model = nn.Sequential(
    nn.Conv2d(1, 16, 5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 4 * 4, 100),
    nn.ReLU(), 
    nn.Linear(100, 10),
)
SCAN_LAYER = 1

classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    input_shape=(1, 28, 28),
    nb_classes=10,
    clip_values=(0.0, 1.0),
)
classifier.fit(x_train, y_train, batch_size=128, nb_epochs=EPOCHS)

bgd_data = x_train[:N_BACKGROUND]
detector = SubsetScanningDetector(classifier, bgd_data, layer=SCAN_LAYER, verbose=False)

# Clean samples and (different) adversarial samples are scored individually.
clean_x = x_test[N_SCAN : 2 * N_SCAN]
pred_clean = np.argmax(classifier.predict(x_test[:N_EXAMPLES]), axis=1)

results = []
for eps in EPS_VALUES:
    x_adv = FastGradientMethod(classifier, eps=eps).generate(x=x_test[:N_SCAN])
    clean_scores, adv_scores, _ = detector.scan(clean_x, x_adv)
    pred_adv = np.argmax(classifier.predict(x_adv[:N_EXAMPLES]), axis=1)
    results.append((eps, x_adv, clean_scores, adv_scores, pred_adv))
    print(f"eps={eps}: mean clean score {clean_scores.mean():.1f}, mean adversarial score {adv_scores.mean():.1f}")

all_scores = np.concatenate([np.concatenate([r[2], r[3]]) for r in results])
bins = np.linspace(all_scores.min(), all_scores.max(), 31)
max_count = max(np.histogram(s, bins=bins)[0].max() for r in results for s in (r[2], r[3]))

frames = []
for eps, x_adv, clean_scores, adv_scores, pred_adv in results:
    fig = plt.figure(figsize=(1.6 * N_EXAMPLES, 7.5))
    grid = fig.add_gridspec(3, N_EXAMPLES, height_ratios=[1, 1, 1.4])
    for j in range(N_EXAMPLES):
        ax_clean = fig.add_subplot(grid[0, j])
        ax_clean.imshow(x_test[j, 0], cmap="gray", vmin=0, vmax=1)
        ax_clean.set_title(f"pred: {pred_clean[j]}", fontsize=9)
        ax_adv = fig.add_subplot(grid[1, j])
        ax_adv.imshow(x_adv[j, 0], cmap="gray", vmin=0, vmax=1)
        ax_adv.set_title(f"pred: {pred_adv[j]}", fontsize=9, color="green" if pred_adv[j] == y_test[j] else "red")
        for ax in (ax_clean, ax_adv):
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel("clean" if ax is ax_clean else f"FGSM eps={eps}")

    ax_hist = fig.add_subplot(grid[2, :])
    ax_hist.hist(clean_scores, bins=bins, alpha=0.6, label="clean")
    ax_hist.hist(adv_scores, bins=bins, alpha=0.6, label="adversarial (FGSM)")
    ax_hist.set_xlim(bins[0], bins[-1])
    ax_hist.set_ylim(0, max_count * 1.1)
    ax_hist.set_xlabel("Subset scanning score")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"eps = {eps}")
    ax_hist.legend()
    fig.tight_layout()

    fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
    plt.close(fig)

frames[0].save(
    "examples/global_attacks_example.gif",
    save_all=True,
    append_images=frames[1:],
    duration=FRAME_DURATION_MS,
    loop=0,
)
print("Saved global_attacks_example.gif")
