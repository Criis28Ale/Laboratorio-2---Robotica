import time
import random
import cv2
import numpy as np
import torch

from torch.utils.data import DataLoader

from config import SEED, BATCH_SIZE, NUM_WORKERS
from dataset import F1Dataset
from model import F1CNN
from loss import WeightedControlLoss


def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    cv2.setNumThreads(0)

    assert torch.cuda.is_available(), "CUDA no disponible"

    device = torch.device("cuda")

    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()

    print("\n=== DISPOSITIVO ===")
    print("GPU:", torch.cuda.get_device_name(0))

    dataset = F1Dataset(
        split="train",
        augment=True
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = F1CNN().to(device)

    criterion = WeightedControlLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    scaler = torch.amp.GradScaler("cuda")

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("\n=== MODELO ===")
    print(model)
    print("Parametros totales:", parameters)

    print("\n=== DATASET ===")
    print("Imagenes:", len(dataset))
    print("Batch size:", BATCH_SIZE)
    print("Batches por epoca:", len(loader))

    model.train()

    torch.cuda.synchronize()
    start = time.perf_counter()

    losses = []

    print("\n=== PRUEBA DE ENTRENAMIENTO ===")

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(
            device, non_blocking=True
        )

        labels = labels.to(
            device, non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):

            predictions = model(images)

            loss = criterion(
                predictions, labels
            )

        assert torch.isfinite(loss).item(), \
            "Loss no finita"

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

        if batch_idx == 0:

            print("Entrada:", images.shape)
            print("Etiquetas:", labels.shape)
            print("Predicciones:", predictions.shape)
            print("Dispositivo:", predictions.device)

        print(
            f"Batch {batch_idx + 1:02d}/20"
            f" | Loss: {loss.item():.6f}"
        )

        if batch_idx + 1 == 20:
            break

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print("\n=== RESULTADOS ===")
    print("Batches procesados:", len(losses))
    print("Imagenes procesadas:", len(losses) * BATCH_SIZE)
    print("Loss promedio:", np.mean(losses))
    print("Loss inicial:", losses[0])
    print("Loss final:", losses[-1])
    print("Tiempo total:", round(elapsed, 2), "segundos")

    print(
        "VRAM maxima asignada:",
        round(
            torch.cuda.max_memory_allocated() / 1024**2,
            2
        ),
        "MB"
    )

    print("\nPrueba de entrenamiento completada.")


if __name__ == "__main__":
    main()
