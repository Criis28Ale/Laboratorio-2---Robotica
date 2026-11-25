import json
import torch
import onnx

from config import (
    MODELS_DIR,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS,
    CROP_TOP,
    V_MIN,
    V_MAX,
    W_SCALE
)

from model import F1CNN


def main():

    model_dir = MODELS_DIR / "f1_baseline"

    checkpoint_path = model_dir / "best_model.pt"

    onnx_path = model_dir / "f1_model.onnx"

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model = F1CNN()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    dummy_input = torch.zeros(
        1,
        IMAGE_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        dtype=torch.float32
    )

    print("\n=== MODELO ORIGINAL ===")
    print("Epoca:", checkpoint["epoch"])
    print("Validation Loss:", checkpoint["best_loss"])

    print(
        "Parametros:",
        sum(p.numel() for p in model.parameters())
    )

    print("Entrada:", dummy_input.shape)

    with torch.inference_mode():

        output = model(dummy_input)

        print("Salida:", output.shape)

    print("\n=== EXPORTACION ONNX ===")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["controls"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "controls": {0: "batch_size"}
        },
        dynamo=False
    )

    print("Archivo generado:", onnx_path)

    print("\n=== VALIDACION ONNX ===")

    onnx_model = onnx.load(str(onnx_path))

    onnx.checker.check_model(onnx_model)

    print("Estructura ONNX valida")

    print(
        "Tamano:",
        round(onnx_path.stat().st_size / 1024**2, 3),
        "MB"
    )

    metadata = {
        "architecture": "F1CNN",
        "best_epoch": int(checkpoint["epoch"]),
        "validation_loss": float(checkpoint["best_loss"]),
        "input_name": "image",
        "output_name": "controls",
        "input_shape": [
            1,
            IMAGE_CHANNELS,
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        ],
        "input_dtype": "float32",
        "color_format": "RGB",
        "layout": "NCHW",
        "original_width": 640,
        "original_height": 480,
        "crop_top": CROP_TOP,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "pixel_normalization": "divide_by_255",
        "output_order": ["v_normalized", "w_normalized"],
        "v_min": V_MIN,
        "v_max": V_MAX,
        "w_scale": W_SCALE,
        "opset": 17
    }

    metadata_path = model_dir / "model_metadata.json"

    with open(metadata_path, "w") as f:

        json.dump(metadata, f, indent=4)

    print("Metadata:", metadata_path)

    print("\nExportacion completada correctamente.")


if __name__ == "__main__":
    main()
