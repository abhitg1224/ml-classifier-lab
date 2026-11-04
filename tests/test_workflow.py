import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.pipeline import Pipeline

from artifacts import file_digest, load_models
from data import FEATURES, LABELS, TARGET, parse_csv
from metrics import evaluate
from models import MODEL_FILES
from test_data import sample_data


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_isolated_cli_training_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source.csv", root / "experiment"
            sample_data().to_csv(source, index=False)
            command = [
                sys.executable, "-W", "error::FutureWarning", str(ROOT / "train.py"),
                "--source", str(source), "--output-dir", str(output),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            split = json.loads((output / "split.json").read_text())
            self.assertEqual((split["training_rows"], split["test_rows"]), (448, 112))
            self.assertEqual(split["feature_overlap"], 0)
            expected = json.loads((output / "results.json").read_text())
            holdout = parse_csv((output / "test_data.csv").read_bytes())
            manifest = json.loads((output / "model" / "manifest.json").read_text())
            self.assertEqual(manifest["test_csv_sha256"], file_digest(output / "test_data.csv"))
            with patch.object(Pipeline, "fit", side_effect=AssertionError("Inference must not fit")):
                for name, model in load_models(output / "model").items():
                    self.assertEqual(evaluate(model, holdout), expected["models"][name])
            self.assertEqual(pd.read_csv(output / "metrics.csv").shape, (5, 7))
            before = {path: file_digest(path) for path in output.rglob("*") if path.is_file()}
            repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("Refusing to overwrite", repeated.stderr)
            self.assertEqual(before, {path: file_digest(path) for path in before})

    def test_conflicting_source_produces_no_experiment_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = sample_data()
            conflict = frame.iloc[:1].copy()
            conflict[TARGET] = LABELS[1]
            source, output = root / "source.csv", root / "experiment"
            pd.concat([frame, conflict], ignore_index=True).to_csv(source, index=False)
            completed = subprocess.run(
                [sys.executable, str(ROOT / "train.py"), "--source", str(source),
                 "--output-dir", str(output)],
                capture_output=True, text=True, timeout=60,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("conflicting", completed.stderr)
            self.assertFalse(output.exists())

    def test_saved_holdout_results_reproduce_without_fitting(self):
        holdout = parse_csv((ROOT / "test_data.csv").read_bytes())
        expected = json.loads((ROOT / "results.json").read_text())["models"]
        with patch.object(Pipeline, "fit", side_effect=AssertionError("Inference must not fit")):
            models = load_models(ROOT / "model")
            self.assertEqual(set(models), set(MODEL_FILES))
            for name, model in models.items():
                self.assertEqual(evaluate(model, holdout), expected[name])
                self.assertEqual(list(model.feature_names_in_), list(FEATURES))
                self.assertEqual(set(model.classes_), set(LABELS))

    def test_manifest_and_any_checksum_fail_before_deserializing(self):
        original = json.loads((ROOT / "model" / "manifest.json").read_text())
        mutations = (
            (lambda item: item.update({"format_version": 2}), "format"),
            (lambda item: item["versions"].update({"scikit-learn": "0"}), "versions differ"),
            (lambda item: item.update({"features": []}), "schema"),
            (lambda item: item["models"]["Logistic Regression"].update(
                {"filename": "../other.joblib"}), "filename"),
            (lambda item: item["models"]["Support Vector Machine"].update(
                {"sha256": "0" * 64}), "checksum"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "model"
            shutil.copytree(ROOT / "model", directory)
            for mutate, message in mutations:
                with self.subTest(message=message):
                    metadata = json.loads(json.dumps(original))
                    mutate(metadata)
                    (directory / "manifest.json").write_text(json.dumps(metadata))
                    with patch("artifacts.joblib.load", side_effect=AssertionError("Must not load")):
                        with self.assertRaisesRegex(ValueError, message):
                            load_models(directory)


if __name__ == "__main__":
    unittest.main()
