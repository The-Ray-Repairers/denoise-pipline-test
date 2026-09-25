import os
import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np

from models import SpatialUNet, RecurrentUNet


def export():
    parser = argparse.ArgumentParser(description="Export Ray Repair Denoiser to ONNX for UE5 / TensorRT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="ray_repair_denoiser.onnx", help="Output .onnx filepath")
    parser.add_argument("--height", type=int, default=1080, help="Default viewport height")
    parser.add_argument("--width", type=int, default=1920, help="Default viewport width")
    parser.add_argument("--dynamic_shapes", action="store_true", default=True, help="Export with dynamic H and W")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_args = checkpoint.get("args", {})

    model_type = saved_args.get("model", "unet")
    in_channels = saved_args.get("in_channels", 7)
    base_channels = saved_args.get("base_channels", 32)
    num_levels = saved_args.get("num_levels", 4)

    if model_type == "unet":
        model = SpatialUNet(
            in_channels=in_channels,
            out_channels=3,
            base_channels=base_channels,
            num_levels=num_levels,
        )
        dummy_input = torch.randn(1, in_channels, args.height, args.width, dtype=torch.float32)
        input_names = ["input_buffers"]
        output_names = ["denoised_radiance"]
        
        dynamic_axes = {
            "input_buffers": {0: "batch", 2: "height", 3: "width"},
            "denoised_radiance": {0: "batch", 2: "height", 3: "width"}
        } if args.dynamic_shapes else None

    else:
        # Recurrent Single-Frame Mode (Frame + Hidden State in, Frame + New Hidden State out)
        model = RecurrentUNet(
            in_channels=in_channels,
            out_channels=3,
            base_channels=base_channels,
            num_levels=num_levels,
        )
        latent_dim = base_channels * (2 ** num_levels)
        latent_h = args.height // (2 ** num_levels)
        latent_w = args.width // (2 ** num_levels)

        dummy_frame = torch.randn(1, in_channels, args.height, args.width, dtype=torch.float32)
        dummy_h = torch.randn(1, latent_dim, latent_h, latent_w, dtype=torch.float32)
        dummy_input = (dummy_frame, dummy_h)
        input_names = ["input_frame", "h_prev"]
        output_names = ["denoised_frame", "h_next"]

        dynamic_axes = {
            "input_frame": {0: "batch", 2: "height", 3: "width"},
            "h_prev": {0: "batch", 2: "latent_h", 3: "latent_w"},
            "denoised_frame": {0: "batch", 2: "height", 3: "width"},
            "h_next": {0: "batch", 2: "latent_h", 3: "latent_w"},
        } if args.dynamic_shapes else None

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Exporting ONNX model to {args.output} (Opset {args.opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Validate ONNX graph
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validated successfully!")

    # Verify with ONNX Runtime
    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    print(f"ONNX Runtime verified. Input nodes: {[inp.name for inp in session.get_inputs()]}")
    print(f"Output nodes: {[out.name for out in session.get_outputs()]}")
    print(f"\n[Ready for UE5 NNE (Neural Network Engine) / TensorRT deployment]")


if __name__ == "__main__":
    export()
