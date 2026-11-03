"""Prepare reproducible training and test partitions from the UCI source."""

import argparse
import json
from pathlib import Path

from data import download_dataset, parse_csv, split_dataset


ROOT = Path(__file__).resolve().parent


def prepare(source: Path, output_dir: Path, *, download: bool = False) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}. Choose a fresh directory.")
    if download:
        frame = download_dataset(source)
    else:
        frame = parse_csv(source.read_bytes(), max_rows=None)
    training, testing, report = split_dataset(frame)
    output_dir.mkdir(parents=True)
    training.to_csv(output_dir / "train_data.csv", index=False, lineterminator="\n", mode="x")
    testing.to_csv(output_dir / "test_data.csv", index=False, lineterminator="\n", mode="x")
    with (output_dir / "split.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "dry_bean.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "prepared")
    parser.add_argument("--download", action="store_true", help="Download the original UCI source.")
    args = parser.parse_args()
    try:
        report = prepare(args.source, args.output_dir, download=args.download)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
