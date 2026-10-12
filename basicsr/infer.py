"""Restore one complete RGB frame sequence."""

import argparse
from pathlib import Path

import cv2
import torch
import yaml

from basicsr.data.paired_video_dataset import frame_paths, read_sequence
from basicsr.utils.checkpoint import load_generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Folder of ordered JPG/PNG frames"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, help="Optional training YAML for nondefault model dimensions"
    )
    parser.add_argument("--param-key", choices=("params", "params_ema"), default="params_ema")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    from basicsr.archs.modal_content_pole_wavelet_dwt3_arch import ModalContentPoleWaveletDWT3

    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires CUDA and MMCV with compiled operators.")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output folder.")
    options = yaml.safe_load(args.config.read_text())["model"] if args.config else {}
    options = {**options, "spynet_pretrained": None}
    model = ModalContentPoleWaveletDWT3(**options).cuda().eval()
    load_generator(model, args.checkpoint, args.param_key)
    model.inference_cpu_offload = args.cpu_offload
    paths = frame_paths(args.input)
    try:
        with torch.no_grad():
            restored = model(read_sequence(paths).unsqueeze(0).cuda())["restored"][0]
    finally:
        model.clear_inference_cache()
    if not torch.isfinite(restored).all():
        raise FloatingPointError("Model output contains nonfinite pixels.")
    args.output.mkdir(parents=True, exist_ok=True)
    for path, frame in zip(paths, restored):
        image = (frame.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        destination = args.output / f"{path.stem}.png"
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Cannot write {destination}")


if __name__ == "__main__":
    main()
