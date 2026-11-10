import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from transformer_lens import TransformerBridge

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Carlos le dio un libro a María porque ella quería leer. ¿Quién quería leer?"
LAYER = 12
OUTPUT_PATH = "examples/attention_example.gif"
FRAME_DURATION_MS = 800

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = TransformerBridge.boot_transformers(MODEL_NAME, device=device)
model.enable_compatibility_mode()
model.eval()

tokens = model.to_str_tokens(PROMPT)
with torch.no_grad():
    _, cache = model.run_with_cache(PROMPT)

# Attention pattern: (batch, n_heads, query_pos, key_pos); each row sums to 1
pattern = cache["pattern", LAYER][0].float().cpu().numpy()
n_heads = pattern.shape[0]
n_tokens = len(tokens)

n_cols = 8
n_rows = (n_heads + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
images = []
boxes = []
for head, ax in enumerate(axes.flat):
    if head >= n_heads:
        ax.axis("off")
        continue
    images.append(ax.imshow(np.full((n_tokens, n_tokens), np.nan), cmap="Blues", vmin=0, vmax=1))
    box = Rectangle((-0.5, -0.5), n_tokens, 1, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(box)
    boxes.append(box)
    ax.set_title(f"L{LAYER} H{head}", fontsize=14)
    ax.set_xticks(range(n_tokens))
    ax.set_yticks(range(n_tokens))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
fig.supxlabel("Key (attended-to token)", fontsize=18)
fig.supylabel("Query (attending token)", fontsize=18)

title = fig.suptitle(" ", fontsize=22)
fig.tight_layout()

# One frame per query token: reveal rows up to the current token and outline it
frames = []
for q in range(n_tokens):
    for head in range(n_heads):
        revealed = np.full((n_tokens, n_tokens), np.nan)
        revealed[: q + 1] = pattern[head, : q + 1]
        images[head].set_data(revealed)
        boxes[head].set_y(q - 0.5)
    title.set_text(f"{MODEL_NAME.split('/')[-1]} layer {LAYER}: query token {q} = {tokens[q]!r}")
    fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
    print(f"Rendered frame {q + 1}/{n_tokens}")
plt.close(fig)

frames[0].save(
    OUTPUT_PATH,
    save_all=True,
    append_images=frames[1:],
    duration=FRAME_DURATION_MS,
    loop=0,
)
print(f"Saved {OUTPUT_PATH}")
