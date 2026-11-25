import time
import json

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import (
    MODELS_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    V_MIN,
    V_MAX,
    W_SCALE
)

from dataset import F1Dataset
from model import F1CNN


def calculate_metrics(real, predicted):

    mse = mean_squared_error(real, predicted)

    return {
        "MAE": float(
            mean_absolute_error(real, predicted)
        ),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(real, predicted))
    }


def denormalize(values):

    result = values.copy()

    result[:, 0] = (
        values[:, 0] * (V_MAX - V_MIN) + V_MIN
    )

    result[:, 1] = values[:, 1] * W_SCALE

    return result


def main():

    cv2.setNumThreads(0)

    assert torch.cuda.is_available(), "CUDA no disponible"

    device = torch.device("cuda")

    output_dir = RESULTS_DIR / "f1_baseline" / "evaluation"

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        MODELS_DIR / "f1_baseline" / "best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model = F1CNN().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\n=== MODELO ===")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Epoca seleccionada:", checkpoint["epoch"])
    print("Validation Loss:", checkpoint["best_loss"])

    dataset = F1Dataset(
        split="test",
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    predictions = []
    targets = []

    print("\n=== EVALUACION ===")
    print("Imagenes de prueba:", len(dataset))

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        for images, labels in tqdm(loader):

            images = images.to(
                device,
                non_blocking=True
            )

            output = model(images)

            predictions.append(
                output.cpu().numpy()
            )

            targets.append(
                labels.numpy()
            )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    predictions = denormalize(
        np.concatenate(predictions)
    )

    targets = denormalize(
        np.concatenate(targets)
    )

    assert np.isfinite(predictions).all()
    assert np.isfinite(targets).all()

    print("\n=== METRICAS DEL MODELO ===")

    metrics = {}

    for index, name in enumerate(["v", "w"]):

        metrics[name] = calculate_metrics(
            targets[:, index],
            predictions[:, index]
        )

        print(f"\nVariable: {name}")

        for key, value in metrics[name].items():
            print(f"{key}: {value:.6f}")

    # Predictor constante basado en entrenamiento

    train_df = pd.read_csv(
        SPLITS_DIR / "train.csv"
    )

    baseline_v = float(train_df["v"].mean())
    baseline_w = float(train_df["w"].mean())

    baseline_predictions = np.tile(
        [baseline_v, baseline_w],
        (len(dataset), 1)
    )

    baseline_metrics = {}

    print("\n=== COMPARACION CONTRA BASELINE ===")

    for index, name in enumerate(["v", "w"]):

        baseline_metrics[name] = calculate_metrics(
            targets[:, index],
            baseline_predictions[:, index]
        )

        improvement = (
            1 - metrics[name]["MAE"]
            / baseline_metrics[name]["MAE"]
        ) * 100

        print(f"\nVariable: {name}")
        print("MAE baseline:",
              baseline_metrics[name]["MAE"])
        print("MAE CNN:", metrics[name]["MAE"])
        print(f"Reduccion del error: {improvement:.2f}%")

    # Guardar predicciones individuales

    test_df = pd.read_csv(
        SPLITS_DIR / "test.csv"
    )

    df = pd.DataFrame({
        "image": test_df["image"],
        "v_real": targets[:, 0],
        "v_pred": predictions[:, 0],
        "w_real": targets[:, 1],
        "w_pred": predictions[:, 1]
    })

    df["error_v"] = df["v_pred"] - df["v_real"]
    df["error_w"] = df["w_pred"] - df["w_real"]

    df["abs_error_v"] = df["error_v"].abs()
    df["abs_error_w"] = df["error_w"].abs()

    # Clasificacion por magnitud del giro real

    magnitude = df["w_real"].abs()

    df["curve_type"] = np.select(
        [
            magnitude <= 0.1,
            magnitude <= 0.5,
            magnitude <= 1.0
        ],
        [
            "Recta",
            "Curva suave",
            "Curva moderada"
        ],
        default="Curva pronunciada"
    )

    curve_metrics = (
        df.groupby("curve_type")
        .agg(
            images=("image", "count"),
            mae_v=("abs_error_v", "mean"),
            mae_w=("abs_error_w", "mean")
        )
        .reset_index()
    )

    print("\n=== RESULTADOS POR TIPO DE CURVA ===")
    print(curve_metrics.to_string(index=False))

    df.to_csv(
        output_dir / "predictions.csv",
        index=False
    )

    curve_metrics.to_csv(
        output_dir / "metrics_by_curve.csv",
        index=False
    )

    # Predicciones fuera del rango de referencia

    outside_v = int(
        (
            (df["v_pred"] < V_MIN)
            | (df["v_pred"] > V_MAX)
        ).sum()
    )

    outside_w = int(
        (df["w_pred"].abs() > W_SCALE).sum()
    )

    print("\n=== LIMITES DE VELOCIDAD ===")
    print("Predicciones v fuera de rango:", outside_v)
    print("Predicciones w fuera de rango:", outside_w)

    # Benchmark de inferencia individual en GPU

    sample = torch.zeros(
        1, 3, 80, 200,
        device=device
    )

    latencies = []

    with torch.inference_mode():

        for _ in range(50):
            model(sample)

        torch.cuda.synchronize()

        for _ in range(300):

            start_inference = time.perf_counter()

            model(sample)

            torch.cuda.synchronize()

            latency = (
                time.perf_counter() - start_inference
            ) * 1000

            latencies.append(latency)

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    print("\n=== RENDIMIENTO ===")
    print(f"Evaluacion completa: {elapsed:.2f} s")
    print(f"Imagenes/s: {len(dataset)/elapsed:.2f}")
    print(f"Latencia media GPU: {mean_latency:.4f} ms")
    print(f"Latencia P95 GPU: {p95_latency:.4f} ms")
    print(f"FPS teoricos del modelo: {1000/mean_latency:.2f}")

    # Guardar resumen

    summary = {
        "best_epoch": int(checkpoint["epoch"]),
        "test_images": len(dataset),
        "model_metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "outside_v": outside_v,
        "outside_w": outside_w,
        "inference_mean_ms": mean_latency,
        "inference_p95_ms": p95_latency,
        "evaluation_seconds": elapsed
    }

    with open(output_dir / "metrics.json", "w") as f:

        json.dump(summary, f, indent=4)

    # Grafica 1: reales vs predicciones

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6)
    )

    for index, name in enumerate(["v", "w"]):

        ax = axes[index]

        real = targets[:, index]
        pred = predictions[:, index]

        low = min(real.min(), pred.min())
        high = max(real.max(), pred.max())

        ax.scatter(
            real, pred,
            s=5, alpha=0.2
        )

        ax.plot(
            [low, high],
            [low, high],
            "r--",
            label="Prediccion ideal"
        )

        ax.set_title(
            f"{name}: R2={metrics[name]['R2']:.4f}"
        )

        ax.set_xlabel("Valor real")
        ax.set_ylabel("Prediccion")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "01_real_vs_predicted.png",
        dpi=180
    )

    plt.close()

    # Grafica 2: distribucion de errores

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5)
    )

    axes[0].hist(
        df["error_v"], bins=60
    )

    axes[0].set_title("Error de velocidad lineal")
    axes[0].set_xlabel("Error (m/s)")

    axes[1].hist(
        df["error_w"], bins=60
    )

    axes[1].set_title("Error de velocidad angular")
    axes[1].set_xlabel("Error (rad/s)")

    for ax in axes:
        ax.axvline(0, color="red", linestyle="--")
        ax.set_ylabel("Frecuencia")
        ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "02_error_distribution.png",
        dpi=180
    )

    plt.close()

    # Grafica 3: comparacion con baseline

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5)
    )

    for index, name in enumerate(["v", "w"]):

        axes[index].bar(
            ["Baseline", "CNN"],
            [
                baseline_metrics[name]["MAE"],
                metrics[name]["MAE"]
            ]
        )

        axes[index].set_title(
            f"Comparacion MAE: {name}"
        )

        axes[index].grid(
            axis="y", alpha=0.3
        )

    plt.tight_layout()

    plt.savefig(
        output_dir / "03_baseline_comparison.png",
        dpi=180
    )

    plt.close()

    # Grafica 4: errores por tipo de curva

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 6)
    )

    for index, name in enumerate(["v", "w"]):

        axes[index].bar(
            curve_metrics["curve_type"],
            curve_metrics[f"mae_{name}"]
        )

        axes[index].set_title(
            f"MAE por tipo de curva: {name}"
        )

        axes[index].tick_params(
            axis="x", rotation=25
        )

        axes[index].grid(
            axis="y", alpha=0.3
        )

    plt.tight_layout()

    plt.savefig(
        output_dir / "04_metrics_by_curve.png",
        dpi=180
    )

    plt.close()

    print("\n=== EVALUACION COMPLETADA ===")
    print("Resultados guardados en:", output_dir)


if __name__ == "__main__":
    main()
