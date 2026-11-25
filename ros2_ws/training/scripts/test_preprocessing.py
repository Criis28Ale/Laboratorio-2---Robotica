import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config import DATASET_DIR, RESULTS_DIR
from preprocessing import (
    preprocess_image,
    normalize_labels,
    denormalize_labels
)

df = pd.read_csv(DATASET_DIR / "splits" / "train.csv")

samples = df.sample(6, random_state=42)

fig, axes = plt.subplots(6, 2, figsize=(12, 18))

for i, (_, row) in enumerate(samples.iterrows()):

    original = cv2.imread(
        str(DATASET_DIR / row["image"])
    )

    original = cv2.cvtColor(
        original, cv2.COLOR_BGR2RGB
    )

    processed = preprocess_image(original)

    labels = normalize_labels(row["v"], row["w"])

    recovered = denormalize_labels(*labels)

    assert np.allclose(
        recovered,
        [row["v"], row["w"]],
        atol=1e-6
    )

    axes[i, 0].imshow(original)

    axes[i, 0].set_title(
        f"Original | v={row['v']:.2f} w={row['w']:.2f}"
    )

    axes[i, 1].imshow(
        np.transpose(processed, (1, 2, 0))
    )

    axes[i, 1].set_title("Preprocesada 200x80")

    axes[i, 0].axis("off")
    axes[i, 1].axis("off")

plt.tight_layout()

output = RESULTS_DIR / "preprocessing_preview.png"

plt.savefig(output, dpi=150)
plt.close()

print("Dimensiones originales:", original.shape)
print("Dimensiones procesadas:", processed.shape)
print("Tipo de dato:", processed.dtype)
print("Pixel minimo:", processed.min())
print("Pixel maximo:", processed.max())
print("Etiquetas normalizadas:", labels)
print("Etiquetas recuperadas:", recovered)
print("Prueba de normalizacion correcta")
print("Imagen guardada en:", output)
