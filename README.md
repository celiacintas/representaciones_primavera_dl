# representaciones_primavera_dl

Snippets para visualizar distintas formas de analizar modelos de deep learning:

- **Grad-CAM** sobre una ResNet-50 (imagenes)
- **Cabezas de atención** de TinyLlama con TransformerLens (lenguaje)

## Setup

El proyecto usa [uv](https://docs.astral.sh/uv/) y Python >= 3.10 (ver `.python-version`).

1. Desde la base del repo:

   ```bash
   uv sync
   ```

## Ejemplos

### Grad-CAM: [`examples/gradcam_example.py`](examples/gradcam_example.py)

```bash
uv run python examples/gradcam_example.py
```

![Ejemplo de Grad-CAM](examples/gradcam_example.gif)

### Cabezas de atención: [`examples/attention_example.py`](examples/attention_example.py)

Carga TinyLlama con TransformerLens.

```bash
uv run python examples/attention_example.py
```

![Ejemplo de atención](examples/attention_example.gif)

Para cambiar el modelo, el prompt o la capa, editar `MODEL_NAME`, `PROMPT` y `LAYER` al inicio del script.
