import os
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DIMENSION = "anti-immigration"
N_STATEMENTS = 50  # half positive, half negative
FIRST_LAYER = 1
LAYER_STEP = 3 
FRAME_DURATION_MS = 1000
DATA_URL = "https://raw.githubusercontent.com/celiacintas/personas-llms-analysis/main/data/anti-immigration_statements.csv"
DATA_PATH = "data/anti-immigration_statements.csv"
OUTPUT_PATH = "examples/hidden_example.gif"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Statements from persona evals (https://github.com/anthropics/evals), as in the notebook
if not os.path.exists(DATA_PATH):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=30) as response, open(DATA_PATH, "wb") as f:
        f.write(response.read())
df = pd.read_csv(DATA_PATH)
df = df[df["persona_dim"] == DIMENSION]
positive = df[df["direction"] == "positive"].statement.to_list()[: N_STATEMENTS // 2]
negative = df[df["direction"] == "negative"].statement.to_list()[: N_STATEMENTS // 2]
statements = positive + negative
labels = ["positive"] * len(positive) + ["negative"] * len(negative)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(device)

# Hidden state of the last token (right after the generation prompt) at every layer
per_statement = []
with torch.no_grad():
    for statement in statements:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": statement}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # newer transformers versions may return a BatchEncoding instead of a tensor
        input_ids = getattr(input_ids, "input_ids", input_ids).to(device)
        outputs = model(input_ids, output_hidden_states=True)
        per_statement.append(torch.stack([h[0, -1, :] for h in outputs.hidden_states]).float().cpu().numpy())
hidden = np.stack(per_statement, axis=1)  # (n_layers + 1, n_statements, hidden_size)
n_layers = hidden.shape[0] - 1
print(f"Hidden states: {hidden.shape}")

palette = {"positive": "tab:blue", "negative": "tab:red"}
frames = []
for layer in range(FIRST_LAYER, n_layers + 1, LAYER_STEP):
    coords = PCA(n_components=2).fit_transform(hidden[layer])
    df_pcs = pd.DataFrame({"PC0": coords[:, 0], "PC1": coords[:, 1], "direction": labels})

    fig, ax = plt.subplots(figsize=(7, 6))
    # 2D KDE contours per class, as in the repo's generate_PCA_directions
    sns.kdeplot(data=df_pcs, x="PC0", y="PC1", hue="direction", palette=palette, levels=5, alpha=0.8, common_norm=False, ax=ax)
    sns.scatterplot(data=df_pcs, x="PC0", y="PC1", hue="direction", palette=palette, s=60, alpha=0.7, ax=ax)
    # kdeplot adds its own legend handles only for contours; rebuild a clean one from the scatter
    handles, texts = ax.get_legend_handles_labels()
    ax.legend(handles[: len(palette)], texts[: len(palette)], title="direction", loc="upper right")
    ax.set_xlabel("PC0")
    ax.set_ylabel("PC1")
    ax.set_title(
        f"{MODEL_NAME.split('/')[-1]} PCA, layer {layer}/{n_layers}\n{DIMENSION} ({len(statements)} statements)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
    plt.close(fig)
    print(f"Rendered layer {layer}/{n_layers}")

frames[0].save(
    OUTPUT_PATH,
    save_all=True,
    append_images=frames[1:],
    duration=FRAME_DURATION_MS,
    loop=0,
)
print(f"Saved {OUTPUT_PATH}")
