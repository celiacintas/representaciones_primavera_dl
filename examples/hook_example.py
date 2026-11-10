import collections
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.datasets import Imagenette
from torchvision.models import AlexNet_Weights, alexnet

# AlexNet conv layers, named as in model.named_modules()
LAYERS = ["features.0", "features.3", "features.6", "features.8", "features.10"]
NUM_IMAGES = 4
CLASS_NAME = "parachute" 
OUTPUT_PATH = "examples/hook_example.png"

weights = AlexNet_Weights.DEFAULT
model = alexnet(weights=weights).eval()

activations = collections.defaultdict(list)


def save_activation(name, module, inputs, output):
    activations[name].append(output.detach().cpu())


# partial assigns the layer name to each hook; keep the handles to remove the hooks later
handles = []
for name, module in model.named_modules():
    if name in LAYERS:
        handles.append(module.register_forward_hook(partial(save_activation, name)))

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ]
)
dataset = Imagenette(root="data", split="val", size="160px", download=True)
class_idx = next(i for i, names in enumerate(dataset.classes) if CLASS_NAME in names)
indices = [i for i, (_, label) in enumerate(dataset._samples) if label == class_idx][:NUM_IMAGES]
images = [dataset[i][0].convert("RGB") for i in indices]

with torch.no_grad():
    for img in images:
        model(preprocess(img).unsqueeze(0))

for handle in handles:
    handle.remove()

# Concatenate the forward passes: (num_images, channels, height, width) per layer
features = {name: torch.cat(outputs, dim=0) for name, outputs in activations.items()}
for name in LAYERS:
    print(f"{name}: {tuple(features[name].shape)}")

fig, axes = plt.subplots(NUM_IMAGES, len(LAYERS) + 1, figsize=(3 * (len(LAYERS) + 1), 3 * NUM_IMAGES))
for row, img in enumerate(images):
    axes[row, 0].imshow(img)
    for col, name in enumerate(LAYERS, start=1):
        axes[row, col].imshow(features[name][row].mean(dim=0), cmap="viridis")
        if row == 0:
            axes[row, col].set_title(f"{name}\n{tuple(features[name].shape[1:])}", fontsize=12)
    if row == 0:
        axes[row, 0].set_title("input", fontsize=12)
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=120)
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
