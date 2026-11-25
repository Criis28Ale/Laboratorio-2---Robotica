import pandas as pd
from config import DATASET_DIR, SPLITS_DIR

df = pd.read_csv(DATASET_DIR / "label.csv")

df["part"] = df["image"].str.split("/").str[0]

df["image_id"] = (
    df["image"]
    .str.extract(r"image_(\d+)\.png$")[0]
    .astype(int)
)

train_parts = []
val_parts = []
test_parts = []
gap_parts = []

for part, group in df.groupby("part"):

    group = group.sort_values("image_id")
    group = group.reset_index(drop=True)

    assert len(group) == 5000, part

    train_parts.append(group.iloc[:3400])
    gap_parts.append(group.iloc[3400:3500])

    val_parts.append(group.iloc[3500:4200])
    gap_parts.append(group.iloc[4200:4300])

    test_parts.append(group.iloc[4300:])

splits = {
    "train": pd.concat(train_parts, ignore_index=True),
    "val": pd.concat(val_parts, ignore_index=True),
    "test": pd.concat(test_parts, ignore_index=True),
    "gap": pd.concat(gap_parts, ignore_index=True)
}

all_paths = set()

for name, data in splits.items():

    paths = set(data["image"])

    assert len(paths) == len(data)
    assert not all_paths.intersection(paths)

    all_paths.update(paths)

    data.to_csv(
        SPLITS_DIR / f"{name}.csv",
        index=False
    )

    print(f"\n=== {name.upper()} ===")
    print("Imagenes:", len(data))
    print("Velocidad media:", data["v"].mean())
    print("Giro medio:", data["w"].mean())
    print("Velocidad igual a 4:",
          round((data["v"] == 4).mean() * 100, 2), "%")

    print("\nImagenes por carpeta:")
    print(data["part"].value_counts().sort_index())

assert len(all_paths) == len(df)

print("\nDivision completada correctamente.")
print("No existen rutas compartidas entre conjuntos.")
print("Total de imagenes:", len(all_paths))
