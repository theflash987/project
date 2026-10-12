"""Evaluate a checkpoint on complete paired video sequences."""

import argparse
from pathlib import Path

import yaml

from basicsr.data.paired_video_dataset import PairedSequenceDataset
from basicsr.utils.checkpoint import load_generator
from basicsr.utils.dist_util import is_distributed, rank_and_size, setup
from basicsr.utils.evaluation import evaluate, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="JSON metric report")
    parser.add_argument("--param-key", choices=("params", "params_ema"), default="params_ema")
    args = parser.parse_args()
    from basicsr.archs.modal_content_pole_wavelet_dwt3_arch import ModalContentPoleWaveletDWT3

    config = yaml.safe_load(args.config.read_text())
    try:
        device = setup()
        model = ModalContentPoleWaveletDWT3(**{**config["model"], "spynet_pretrained": None}).to(
            device
        )
        load_generator(model, args.checkpoint, args.param_key)
        dataset = PairedSequenceDataset(config["datasets"]["val"])
        report = evaluate(model, dataset, device, config["validation"]["cpu_offload"])
        if rank_and_size()[0] == 0:
            write_report(report, args.output)
            print(report["final"]["sequence_macro"])
    finally:
        if is_distributed():
            import torch

            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
