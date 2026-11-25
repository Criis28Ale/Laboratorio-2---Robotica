import cv2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from config import DATASET_DIR, SPLITS_DIR
from preprocessing import (
    preprocess_image,
    normalize_labels
)


class F1Dataset(Dataset):

    def __init__(self, split="train", augment=False):

        self.df = pd.read_csv(
            SPLITS_DIR / f"{split}.csv"
        )

        self.augment = augment and split == "train"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        path = DATASET_DIR / row["image"]

        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(
                f"No se pudo cargar: {path}"
            )

        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        v = float(row["v"])
        w = float(row["w"])

        if self.augment:

            # Reflejo horizontal
            if np.random.random() < 0.5:
                image = cv2.flip(image, 1)
                w = -w

            # Brillo y contraste moderados
            if np.random.random() < 0.3:

                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.uniform(-10, 10)

                image = np.clip(
                    image.astype(np.float32)
                    * alpha + beta,
                    0, 255
                ).astype(np.uint8)

        image = preprocess_image(image)
        labels = normalize_labels(v, w)

        return (
            torch.from_numpy(image),
            torch.from_numpy(labels)
        )
