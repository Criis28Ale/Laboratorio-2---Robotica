import time
import json
import numpy as np
import torch
import onnxruntime as ort

from config import (
    MODELS_DIR,
    RESULTS_DIR,
    V_MIN,
    V_MAX,
    W_SCALE
)

from model import F1CNN
from dataset import F1Dataset


def main():

    model_dir = MODELS_DIR / "f1_baseline"

    output_dir = (
        RESULTS_DIR / "f1_baseline" / "onnx_validation"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        model_dir / "best_model.pt",
        map_location="cpu",
        weights_only=True
    )

    model = F1CNN()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    dataset = F1Dataset(
        split="test",
        augment=False
    )

    rng = np.random.default_rng(42)

    indices = rng.choice(
        len(dataset),
        size=100,
        replace=False
    )

    images = torch.stack([
        dataset[int(i)][0]
        for i in indices
    ])

    input_array = images.numpy()

    with torch.inference_mode():

        pytorch_output = model(images).numpy()

    available = ort.get_available_providers()

    print("\n=== PROVIDERS DISPONIBLES ===")
    print(available)

    # CPU como referencia numerica

    cpu_session = ort.InferenceSession(
        str(model_dir / "f1_model.onnx"),
        providers=["CPUExecutionProvider"]
    )

    input_name = cpu_session.get_inputs()[0].name

    cpu_output = cpu_session.run(
        None,
        {input_name: input_array}
    )[0]

    print("\n=== EQUIVALENCIA CPU ===")

    print("PyTorch shape:", pytorch_output.shape)
    print("ONNX shape:", cpu_output.shape)

    difference = np.abs(
        pytorch_output - cpu_output
    )

    cpu_max = float(difference.max())
    cpu_mean = float(difference.mean())

    cpu_equivalent = bool(
        np.allclose(
            pytorch_output,
            cpu_output,
            atol=1e-5,
            rtol=1e-4
        )
    )

    print("Diferencia maxima:", cpu_max)
    print("Diferencia media:", cpu_mean)
    print("Equivalencia:", cpu_equivalent)

    summary = {
        "samples": len(indices),
        "cpu_max_difference": cpu_max,
        "cpu_mean_difference": cpu_mean,
        "cpu_equivalent": cpu_equivalent,
        "cuda_equivalent": None
    }

    session = cpu_session

    if "CUDAExecutionProvider" in available:

        try:

            # Precargar librerias CUDA de PyTorch
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()

            gpu_session = ort.InferenceSession(
                str(model_dir / "f1_model.onnx"),
                providers=[
                    (
                        "CUDAExecutionProvider",
                        {"use_tf32": "0"}
                    ),
                    "CPUExecutionProvider"
                ]
            )

            if (
                "CUDAExecutionProvider"
                not in gpu_session.get_providers()
            ):
                raise RuntimeError(
                    "ONNX Runtime no activo CUDA"
                )

            gpu_output = gpu_session.run(
                None,
                {input_name: input_array}
            )[0]

            gpu_difference = np.abs(
                pytorch_output - gpu_output
            )

            gpu_equivalent = bool(
                np.allclose(
                    pytorch_output,
                    gpu_output,
                    atol=1e-5,
                    rtol=1e-4
                )
            )

            print("\n=== EQUIVALENCIA CUDA ===")

            print(
                "Diferencia maxima:",
                float(gpu_difference.max())
            )

            print(
                "Diferencia media:",
                float(gpu_difference.mean())
            )

            print("Equivalencia:", gpu_equivalent)

            summary["cuda_equivalent"] = gpu_equivalent

            summary["cuda_max_difference"] = float(
                gpu_difference.max()
            )

            if gpu_equivalent:
                session = gpu_session

        except Exception as error:

            print("\nCUDA ONNX no disponible:")
            print(error)
            print("Continuando con CPU")

    print("\n=== PREDICCIONES DE EJEMPLO ===")

    for i in range(5):

        v = (
            cpu_output[i, 0]
            * (V_MAX - V_MIN)
            + V_MIN
        )

        w = cpu_output[i, 1] * W_SCALE

        print(
            f"Muestra {i+1}: "
            f"v={v:.4f} m/s | "
            f"w={w:.4f} rad/s"
        )

    print("\n=== BENCHMARK ONNX ===")

    sample = input_array[:1]

    for _ in range(50):

        session.run(
            None,
            {input_name: sample}
        )

    latencies = []

    for _ in range(300):

        start = time.perf_counter()

        session.run(
            None,
            {input_name: sample}
        )

        latencies.append(
            (time.perf_counter() - start) * 1000
        )

    mean_latency = float(np.mean(latencies))

    p95_latency = float(
        np.percentile(latencies, 95)
    )

    print("Providers:", session.get_providers())

    print(f"Latencia media: {mean_latency:.4f} ms")
    print(f"Latencia P95: {p95_latency:.4f} ms")
    print(f"FPS teoricos: {1000/mean_latency:.2f}")

    summary["providers"] = session.get_providers()
    summary["mean_latency_ms"] = mean_latency
    summary["p95_latency_ms"] = p95_latency

    with open(output_dir / "validation.json", "w") as f:

        json.dump(summary, f, indent=4)

    if not cpu_equivalent:

        raise RuntimeError(
            "El modelo ONNX no es equivalente a PyTorch"
        )

    print("\nValidacion ONNX completada.")
    print("Resultados:", output_dir)


if __name__ == "__main__":
    main()
