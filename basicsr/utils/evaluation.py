"""Whole-video RGB PSNR/SSIM, using the original prediction quantization."""

import json
from pathlib import Path

import torch

from basicsr.utils.dist_util import is_distributed, rank_and_size


def quantize_prediction(value):
    if not torch.isfinite(value).all():
        raise FloatingPointError("Model output contains nonfinite pixels before quantization.")
    value = value.squeeze(0).float().clamp(0.0, 1.0)
    value = (value * 255).round().to(torch.uint8)
    return value.to(torch.float32).div_(255).unsqueeze(0)


def evaluate(model, dataset, device, cpu_offload=True):
    from pyiqa import create_metric

    with torch.backends.cudnn.flags(
        enabled=torch.backends.cudnn.enabled,
        benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.backends.cudnn.deterministic,
        allow_tf32=torch.backends.cudnn.allow_tf32,
    ):
        rank, world_size = rank_and_size()
        model.eval()
        model.inference_cpu_offload = cpu_offload
        names = ("psnr", "ssim")
        metrics = {name: create_metric(name, metric_mode="FR", device=device) for name in names}
        scores = {
            kind: {
                folder: torch.zeros(len(dataset.paths[folder][0]), len(names), device=device)
                for folder in dataset.folders
            }
            for kind in ("final", "base")
        }
        with torch.no_grad():
            for index in range(rank, len(dataset), world_size):
                data = dataset[index]
                try:
                    outputs = model(data["lq"].unsqueeze(0).to(device))
                    for frame in range(data["gt"].shape[0]):
                        gt = data["gt"][frame].unsqueeze(0)
                        for kind, key in (("final", "restored"), ("base", "base_restored")):
                            prediction = quantize_prediction(outputs[key][0, frame].cpu())
                            for metric_index, name in enumerate(names):
                                scores[kind][data["folder"]][frame, metric_index] = metrics[name](
                                    prediction, gt
                                ).item()
                finally:
                    model.clear_inference_cache()
                del outputs, data
        if is_distributed():
            for folders in scores.values():
                for values in folders.values():
                    torch.distributed.all_reduce(values)
        if rank != 0:
            return None
        result = {}
        for kind, folders in scores.items():
            sequence_means = {folder: values.mean(dim=0) for folder, values in folders.items()}
            macro = torch.stack(list(sequence_means.values())).mean(dim=0)
            result[kind] = {
                "sequence_macro": {name: macro[i].item() for i, name in enumerate(names)},
                "per_sequence": {
                    folder: {name: values[i].item() for i, name in enumerate(names)}
                    for folder, values in sequence_means.items()
                },
                "per_frame": {folder: values.cpu().tolist() for folder, values in folders.items()},
            }
        return result


def write_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
