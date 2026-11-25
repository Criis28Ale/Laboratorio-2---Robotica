from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
RESULTS = ROOT / "results" / "eda"

RESULTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET / "label.csv")

sns.set_theme(style="whitegrid")

# 1. Verificacion de imagenes

print("\n=== VERIFICACION DE IMAGENES ===")

missing = []
corrupted = []
resolutions = Counter()

for image_path in tqdm(df["image"]):
    path = DATASET / image_path

    if not path.is_file():
        missing.append(image_path)
        continue

    image = cv2.imread(str(path))

    if image is None:
        corrupted.append(image_path)
        continue

    h, w, c = image.shape
    resolutions[(w, h, c)] += 1

print("Imagenes faltantes:", len(missing))
print("Imagenes corruptas:", len(corrupted))
print("Resoluciones:", dict(resolutions))
print("Rutas duplicadas:", df["image"].duplicated().sum())
print("Etiquetas no finitas:",
      (~np.isfinite(df[["v", "w"]])).sum().to_dict())

pd.DataFrame(
    {"image": missing}
).to_csv(RESULTS / "missing_images.csv", index=False)

pd.DataFrame(
    {"image": corrupted}
).to_csv(RESULTS / "corrupted_images.csv", index=False)

# 2. Estadisticas generales

print("\n=== DISTRIBUCION DE VELOCIDADES ===")

print("Velocidad igual a 4 m/s:",
      round((df["v"] == 4).mean() * 100, 2), "%")

print("Giros positivos:",
      round((df["w"] > 0).mean() * 100, 2), "%")

print("Giros negativos:",
      round((df["w"] < 0).mean() * 100, 2), "%")

print("Velocidad angular cercana a cero:",
      round((df["w"].abs() < 0.1).mean() * 100, 2), "%")

print("\nCorrelacion:")
print(df[["v", "w"]].corr())

print("\nCorrelacion entre v y magnitud de w:")
print(df["v"].corr(df["w"].abs()))

df[["v", "w"]].describe().to_csv(
    RESULTS / "statistics.csv"
)

# 3. Histogramas

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(
    df["v"], bins=40,
    color="royalblue", ax=axes[0]
)

axes[0].set_title("Distribucion de velocidad lineal")
axes[0].set_xlabel("Velocidad lineal (m/s)")
axes[0].set_ylabel("Frecuencia")

sns.histplot(
    df["w"], bins=60,
    color="darkorange", ax=axes[1]
)

axes[1].set_title("Distribucion de velocidad angular")
axes[1].set_xlabel("Velocidad angular (rad/s)")
axes[1].set_ylabel("Frecuencia")

plt.tight_layout()
plt.savefig(
    RESULTS / "01_distribucion_velocidades.png",
    dpi=200
)
plt.close()

# 4. Relacion entre velocidades

sample = df.sample(5000, random_state=42)

plt.figure(figsize=(9, 6))

plt.scatter(
    sample["w"],
    sample["v"],
    alpha=0.2,
    s=10
)

plt.xlabel("Velocidad angular (rad/s)")
plt.ylabel("Velocidad lineal (m/s)")
plt.title("Relacion entre velocidad lineal y angular")

plt.tight_layout()
plt.savefig(
    RESULTS / "02_relacion_velocidades.png",
    dpi=200
)
plt.close()

# 5. Matriz de correlacion

plt.figure(figsize=(6, 5))

sns.heatmap(
    df[["v", "w"]].corr(),
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".4f"
)

plt.title("Matriz de correlacion")

plt.tight_layout()
plt.savefig(
    RESULTS / "03_correlacion.png",
    dpi=200
)
plt.close()

# 6. Analisis por carpeta

df["part"] = df["image"].str.split("/").str[0]

summary = df.groupby("part").agg(
    images=("image", "count"),
    v_mean=("v", "mean"),
    v_std=("v", "std"),
    w_mean=("w", "mean"),
    w_std=("w", "std")
)

print("\n=== ANALISIS POR CARPETA ===")
print(summary.to_string())

summary.to_csv(RESULTS / "statistics_by_part.csv")

plt.figure(figsize=(12, 5))

summary["images"].plot(
    kind="bar",
    color="royalblue"
)

plt.title("Cantidad de imagenes por carpeta")
plt.ylabel("Cantidad de imagenes")
plt.xlabel("Carpeta")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(
    RESULTS / "04_imagenes_por_carpeta.png",
    dpi=200
)
plt.close()

# 7. Visualizacion de imagenes aleatorias

samples = df.sample(12, random_state=42)

fig, axes = plt.subplots(3, 4, figsize=(16, 10))

for ax, (_, row) in zip(axes.flat, samples.iterrows()):

    image = cv2.imread(str(DATASET / row["image"]))

    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)

    ax.set_title(
        f"v={row['v']:.2f} | w={row['w']:.2f}"
    )

    ax.axis("off")

plt.suptitle("Muestras del dataset de conduccion")
plt.tight_layout()

plt.savefig(
    RESULTS / "05_muestras_dataset.png",
    dpi=200
)
plt.close()

# 8. Visualizacion por direccion de giro

groups = [
    ("Giro derecha", df[df["w"] < -0.5]),
    ("Trayectoria recta", df[df["w"].abs() < 0.1]),
    ("Giro izquierda", df[df["w"] > 0.5])
]

fig, axes = plt.subplots(3, 4, figsize=(16, 10))

for i, (name, group) in enumerate(groups):

    selected = group.sample(
        min(4, len(group)),
        random_state=42
    )

    for ax in axes[i]:
        ax.axis("off")

    for j, (_, row) in enumerate(selected.iterrows()):

        ax = axes[i, j]
        image = cv2.imread(str(DATASET / row["image"]))

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)

        ax.set_title(
            f"{name}\nv={row['v']:.2f} | w={row['w']:.2f}"
        )

plt.tight_layout()

plt.savefig(
    RESULTS / "06_direcciones_conduccion.png",
    dpi=200
)
plt.close()

print("\n=== ANALISIS COMPLETADO ===")
print("Resultados guardados en:", RESULTS)
