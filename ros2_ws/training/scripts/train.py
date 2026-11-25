import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import (
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    MODELS_DIR,
    RESULTS_DIR,
    V_MIN,
    V_MAX,
    W_SCALE
)

from dataset import F1Dataset
from model import F1CNN
from loss import WeightedControlLoss


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None
):

    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_samples = 0

    absolute_error = np.zeros(2)
    squared_error = np.zeros(2)

    scales = torch.tensor(
        [V_MAX - V_MIN, W_SCALE],
        device=device
    )

    progress = tqdm(
        loader,
        desc="Train" if training else "Validation",
        leave=False
    )

    for images, labels in progress:

        images = images.to(
            device, non_blocking=True
        )

        labels = labels.to(
            device, non_blocking=True
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            with torch.amp.autocast("cuda"):

                predictions = model(images)

                loss = criterion(
                    predictions, labels
                )

            if not torch.isfinite(loss).item():
                raise RuntimeError("Loss no finita")

            if training:

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        errors = (
            predictions.detach().float()
            - labels.float()
        ) * scales

        absolute_error += (
            errors.abs().sum(dim=0).cpu().numpy()
        )

        squared_error += (
            errors.square().sum(dim=0).cpu().numpy()
        )

        progress.set_postfix(
            loss=f"{loss.item():.5f}"
        )

    mae = absolute_error / total_samples

    rmse = np.sqrt(
        squared_error / total_samples
    )

    return {
        "loss": total_loss / total_samples,
        "mae_v": float(mae[0]),
        "mae_w": float(mae[1]),
        "rmse_v": float(rmse[0]),
        "rmse_w": float(rmse[1])
    }


def save_plots(history, output_dir):

    df = pd.DataFrame(history)

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10)
    )

    axes[0, 0].plot(
        df["epoch"], df["train_loss"],
        label="Train"
    )

    axes[0, 0].plot(
        df["epoch"], df["val_loss"],
        label="Validation"
    )

    axes[0, 0].set_title("Training and validation loss")
    axes[0, 0].legend()

    axes[0, 1].plot(
        df["epoch"], df["train_mae_v"],
        label="Train"
    )

    axes[0, 1].plot(
        df["epoch"], df["val_mae_v"],
        label="Validation"
    )

    axes[0, 1].set_title("MAE linear velocity (m/s)")
    axes[0, 1].legend()

    axes[1, 0].plot(
        df["epoch"], df["train_mae_w"],
        label="Train"
    )

    axes[1, 0].plot(
        df["epoch"], df["val_mae_w"],
        label="Validation"
    )

    axes[1, 0].set_title("MAE angular velocity (rad/s)")
    axes[1, 0].legend()

    axes[1, 1].plot(
        df["epoch"], df["learning_rate"]
    )

    axes[1, 1].set_title("Learning rate")
    axes[1, 1].set_yscale("log")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "training_curves.png",
        dpi=180
    )

    plt.close()


def save_checkpoint(state, path):

    temporary = path.with_suffix(".tmp")

    torch.save(state, temporary)

    temporary.replace(path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=EPOCHS
    )

    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE
    )

    parser.add_argument(
        "--run-name", type=str, default="f1_baseline"
    )

    parser.add_argument(
        "--resume", action="store_true"
    )

    args = parser.parse_args()

    set_seed(SEED)

    cv2.setNumThreads(0)

    assert torch.cuda.is_available(), "CUDA no disponible"

    device = torch.device("cuda")

    torch.backends.cudnn.benchmark = True

    model_dir = MODELS_DIR / args.run_name
    output_dir = RESULTS_DIR / args.run_name

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_path = model_dir / "best_model.pt"
    last_path = model_dir / "last_checkpoint.pt"

    if not args.resume and last_path.exists():
        raise RuntimeError(
            "Ya existe un entrenamiento. "
            "Usa --resume o cambia --run-name."
        )

    train_dataset = F1Dataset(
        split="train", augment=True
    )

    val_dataset = F1Dataset(
        split="val", augment=False
    )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": NUM_WORKERS > 0
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_options
    )

    model = F1CNN().to(device)

    criterion = WeightedControlLoss(
        v_weight=0.4,
        w_weight=0.6
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    scaler = torch.amp.GradScaler("cuda")

    history = []

    start_epoch = 1
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    if args.resume:

        if not last_path.exists():
            raise FileNotFoundError(last_path)

        checkpoint = torch.load(
            last_path,
            map_location=device,
            weights_only=True
        )

        if checkpoint["batch_size"] != args.batch_size:
            raise ValueError(
                "El batch size debe coincidir "
                "con el entrenamiento anterior"
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        history = checkpoint["history"]

        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        best_epoch = checkpoint["best_epoch"]

        epochs_without_improvement = checkpoint[
            "epochs_without_improvement"
        ]

        print("Reanudando desde epoca:", start_epoch)

    writer = SummaryWriter(
        log_dir=str(output_dir / "tensorboard"),
        purge_step=start_epoch if args.resume else None
    )

    print("\n=== CONFIGURACION ===")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Model parameters:",
          sum(p.numel() for p in model.parameters()))

    print("\n=== INICIO DEL ENTRENAMIENTO ===")

    training_start = time.perf_counter()

    try:

        for epoch in range(start_epoch, args.epochs + 1):

            epoch_start = time.perf_counter()

            torch.cuda.reset_peak_memory_stats()

            learning_rate = optimizer.param_groups[0]["lr"]

            print(f"\nEpoch {epoch}/{args.epochs}")

            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                scaler
            )

            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device
            )

            scheduler.step(val_metrics["loss"])

            elapsed = time.perf_counter() - epoch_start

            record = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "time_seconds": elapsed,
                "vram_mb": (
                    torch.cuda.max_memory_allocated()
                    / 1024**2
                )
            }

            for key, value in train_metrics.items():
                record[f"train_{key}"] = value

            for key, value in val_metrics.items():
                record[f"val_{key}"] = value

            history.append(record)

            for key, value in record.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)

            improved = val_metrics["loss"] < best_loss

            if improved:

                best_loss = val_metrics["loss"]
                best_epoch = epoch
                epochs_without_improvement = 0

            else:

                epochs_without_improvement += 1

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "history": history,
                "batch_size": args.batch_size,
                "epochs_without_improvement":
                    epochs_without_improvement
            }

            if improved:

                save_checkpoint(state, best_path)

                print("Nuevo mejor modelo guardado.")

            save_checkpoint(state, last_path)

            pd.DataFrame(history).to_csv(
                output_dir / "history.csv",
                index=False
            )

            writer.flush()

            print(
                f"Train Loss: {train_metrics['loss']:.6f}"
            )

            print(
                f"Val Loss: {val_metrics['loss']:.6f}"
            )

            print(
                f"Val MAE v: {val_metrics['mae_v']:.4f} m/s"
            )

            print(
                f"Val MAE w: {val_metrics['mae_w']:.4f} rad/s"
            )

            print(
                f"Val RMSE v: {val_metrics['rmse_v']:.4f} m/s"
            )

            print(
                f"Val RMSE w: {val_metrics['rmse_w']:.4f} rad/s"
            )

            print(f"LR: {learning_rate:.8f}")
            print(f"Tiempo: {elapsed:.2f} segundos")

            print(
                f"VRAM: {record['vram_mb']:.2f} MB"
            )

            print(
                "Epocas sin mejora:",
                epochs_without_improvement
            )

            if epochs_without_improvement >= PATIENCE:

                print("\nEarly stopping activado.")
                break

    except KeyboardInterrupt:

        print("\nEntrenamiento interrumpido.")
        print("Se conserva la ultima epoca completada.")

    finally:

        writer.close()

        if history:
            save_plots(history, output_dir)

    print("\n=== RESULTADOS ===")

    print("Mejor epoca:", best_epoch)
    print("Mejor validation loss:", best_loss)

    print(
        "Tiempo de esta sesion:",
        round(
            (time.perf_counter() - training_start) / 60,
            2
        ),
        "minutos"
    )

    print("Mejor modelo:", best_path)
    print("Ultimo checkpoint:", last_path)
    print("Resultados:", output_dir)


if __name__ == "__main__":
    main()
