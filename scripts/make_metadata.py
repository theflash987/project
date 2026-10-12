"""Build four-field metadata from paired, consecutively numbered frame folders."""

import argparse
from pathlib import Path

from basicsr.data.paired_video_dataset import PairedSequenceDataset, read_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lq", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dataset = PairedSequenceDataset({"lq": args.lq, "gt": args.gt})
    rows = []
    for folder in dataset.folders:
        paths, _ = dataset.paths[folder]
        indices = [int(path.stem) for path in paths]
        first = indices[0]
        if indices != list(range(first, first + len(paths))):
            raise ValueError(f"Frames in {folder} must be numbered consecutively.")
        shape = read_image(paths[0]).shape
        rows.append(f"{folder} {len(paths)} ({shape[0]},{shape[1]},3) {first}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
