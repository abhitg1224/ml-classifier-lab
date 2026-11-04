"""Run training-only model comparison, save pipelines, and evaluate the holdout."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np

from artifacts import file_digest, load_models, runtime_versions
from data import (
    FEATURES, LABELS, SOURCE_URL, TARGET, download_dataset, parse_csv, split_dataset,
)
from metrics import comparison_table, evaluate
from models import MODEL_FILES, compare_training_folds, make_models


ROOT = Path(__file__).resolve().parent
OUTPUTS = ("model", "test_data.csv", "split.json", "cross_validation.json", "results.json", "metrics.csv")


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def train_experiment(source: Path, output: Path, *, download: bool = False) -> dict:
    existing = [str(output / name) for name in OUTPUTS if (output / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite experiment outputs: "
            + ", ".join(existing) + ". Choose a fresh --output-dir."
        )
    frame = download_dataset(source) if download else parse_csv(source.read_bytes(), max_rows=None)
    training, holdout, split = split_dataset(frame)
    print(
        f"Source={split['source_rows']}; duplicates removed={split['duplicates_removed']}; "
        f"training={len(training)}; holdout={len(holdout)}; feature overlap=0",
        flush=True,
    )
    print("Comparing all five classifiers on training-only folds...", flush=True)
    validation = compare_training_folds(training)
    print(f"Validation-selected classifier: {validation['selected_model']}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    directory = output / "model"
    directory.mkdir()
    holdout.to_csv(output / "test_data.csv", index=False, lineterminator="\n", mode="x")
    saved_holdout = parse_csv((output / "test_data.csv").read_bytes())
    write_json(output / "split.json", split)
    write_json(output / "cross_validation.json", validation)
    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "source_sha256": split["source_sha256"],
        "test_csv_sha256": file_digest(output / "test_data.csv"),
        "features": list(FEATURES),
        "labels": list(LABELS),
        "versions": runtime_versions(),
        "selected_model": validation["selected_model"],
        "models": {},
    }
    fitted = make_models()
    inputs = training.loc[:, list(FEATURES)]
    for name, model in fitted.items():
        print(f"Fitting and saving {name}...", flush=True)
        model.fit(inputs, training[TARGET])
        filename = MODEL_FILES[name]
        joblib.dump(model, directory / filename, compress=3)
        manifest["models"][name] = {
            "filename": filename,
            "sha256": file_digest(directory / filename),
        }
    write_json(directory / "manifest.json", manifest)

    restored = load_models(directory)
    results = {}
    test_inputs = saved_holdout.loc[:, list(FEATURES)]
    for name, model in restored.items():
        if not np.array_equal(model.predict(test_inputs), fitted[name].predict(test_inputs)):
            raise ValueError(f"Predictions changed after reloading {name}.")
        if not np.allclose(
            model.predict_proba(test_inputs), fitted[name].predict_proba(test_inputs),
            rtol=0, atol=1e-12,
        ):
            raise ValueError(f"Probabilities changed after reloading {name}.")
        results[name] = evaluate(model, saved_holdout)
    report = {
        "holdout_rows": len(saved_holdout),
        "selected_using": validation["selection_metric"],
        "selected_model": validation["selected_model"],
        "macro_labels": list(LABELS),
        "auc_method": "macro one-vs-rest using class-aligned probabilities",
        "models": results,
    }
    write_json(output / "results.json", report)
    table = comparison_table(results)
    table.to_csv(output / "metrics.csv", lineterminator="\n", mode="x")
    print(table.to_string(float_format=lambda value: f"{value:.6f}"), flush=True)
    print(f"Verified experiment saved to {output.resolve()}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "dry_bean.csv")
    parser.add_argument("--download", action="store_true", help="Download the original UCI source.")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        train_experiment(args.source, args.output_dir, download=args.download)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Experiment failed: {error}\n")


if __name__ == "__main__":
    main()
