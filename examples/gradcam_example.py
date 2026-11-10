import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from torchvision.datasets import Imagenette
from torchvision.models import ResNet50_Weights, resnet50

NUM_IMAGES = 20
FRAME_DURATION_MS = 1500

IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights).eval().to(device)

layers = {
    "layer1": model.layer1[-1],
    "layer2": model.layer2[-1],
    "layer3": model.layer3[-1],
    "layer4": model.layer4[-1],
}

methods = {
    "GradCAM": GradCAM,
    "GradCAM++": GradCAMPlusPlus,
    "XGradCAM": XGradCAM,
}

resize_crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
normalize = transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
dataset = Imagenette(root="data", split="val", size="160px", download=True)

random.seed(0)
indices = random.sample(range(len(dataset)), NUM_IMAGES)

frames = []
for i in indices:
    img, label = dataset[i]
    rgb_tensor = transforms.functional.to_tensor(resize_crop(img.convert("RGB")))  # (3, H, W) in [0, 1]
    rgb_img = rgb_tensor.permute(1, 2, 0).numpy()
    input_tensor = normalize(rgb_tensor).unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(IMAGENETTE_TO_IMAGENET[label])]

    # Same image: original on the left, then rows = layers, columns = methods
    fig = plt.figure(figsize=(10, 9.5))
    grid = fig.add_gridspec(len(layers), len(methods) + 1)
    ax_orig = fig.add_subplot(grid[:, 0])
    ax_orig.imshow(rgb_img)
    ax_orig.set_title(dataset.classes[label][0])
    ax_orig.axis("off")
    for row, (layer_name, layer) in enumerate(layers.items()):
        for col, (method_name, method_cls) in enumerate(methods.items(), start=1):
            ax = fig.add_subplot(grid[row, col])
            with method_cls(model=model, target_layers=[layer]) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            ax.imshow(show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(method_name)
            if col == 1:
                ax.set_ylabel(layer_name)
    fig.tight_layout()

    fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
    plt.close(fig)
    print(f"Processed image {i} ({dataset.classes[label][0]})")

frames[0].save(
    "examples/gradcam_example.gif",
    save_all=True,
    append_images=frames[1:],
    duration=FRAME_DURATION_MS,
    loop=0,
)
print("Saved gradcam_example.gif")
