from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd

from data import (
    COLUMNS, FEATURES, LABELS, MAX_CSV_BYTES, TARGET,
    download_dataset, parse_csv, split_dataset, validate_data,
)
from prepare_data import prepare


def sample_data(rows_per_class=80):
    rng = np.random.default_rng(42)
    labels = np.repeat(LABELS, rows_per_class)
    class_centers = np.repeat(np.arange(len(LABELS)), rows_per_class)
    features = rng.normal(scale=0.2, size=(len(labels), len(FEATURES)))
    features += class_centers[:, None]
    frame = pd.DataFrame(features, columns=FEATURES)
    frame[TARGET] = labels
    return frame


class DataTests(unittest.TestCase):
    def test_duplicate_safe_split_is_stratified_and_reproducible(self):
        source = sample_data()
        source = pd.concat([source, source.iloc[:12]], ignore_index=True)
        training, testing, report = split_dataset(source)
        again_training, again_testing, again_report = split_dataset(source)
        pd.testing.assert_frame_equal(training, again_training)
        pd.testing.assert_frame_equal(testing, again_testing)
        self.assertEqual(report, again_report)
        self.assertEqual(report["duplicates_removed"], 12)
        self.assertEqual((len(training), len(testing)), (448, 112))
        self.assertEqual(set(training[TARGET]), set(LABELS))
        self.assertEqual(set(testing[TARGET]), set(LABELS))
        self.assertTrue(training.merge(testing, on=list(FEATURES)).empty)
        self.assertTrue(all(counts["test"] == 16 for counts in report["class_counts"].values()))

    def test_invalid_training_sources_are_rejected(self):
        source = sample_data()
        conflict = source.iloc[:1].copy()
        conflict[TARGET] = LABELS[1]
        cases = (
            (source.iloc[:20], "at least 500"),
            (source[source[TARGET] != LABELS[-1]], "at least 500|all seven"),
            (pd.concat([source, conflict], ignore_index=True), "conflicting"),
            (pd.concat([sample_data(1)] * 80, ignore_index=True), "500 unique"),
        )
        for frame, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    split_dataset(frame)

    def test_csv_accepts_reordered_columns_and_utf8_bom(self):
        source = sample_data(2)
        payload = source.loc[:, list(reversed(COLUMNS))].to_csv(index=False).encode("utf-8-sig")
        parsed = parse_csv(payload)
        pd.testing.assert_frame_equal(parsed, source, check_exact=False)

    def test_csv_schema_encoding_and_record_errors_are_explicit(self):
        header = ",".join(COLUMNS)
        cases = (
            (b"", "header"),
            ((header + "\n").encode(), "At least one"),
            (b"Area,Class\n12,SIRA\n", "Incorrect columns"),
            ((header + ",Area\n").encode(), "Duplicate"),
            ((header + "\n1,2\n").encode(), "field count"),
            ((header + '\n"unfinished').encode(), "Malformed"),
            (b"\xff", "UTF-8"),
            (b"x" * (MAX_CSV_BYTES + 1), "10 MiB"),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    parse_csv(payload)

    def test_invalid_values_and_row_limits_are_rejected(self):
        source = sample_data(2).astype(str)
        cases = (
            (FEATURES[0], "text", "numeric"),
            (FEATURES[0], "inf", "finite"),
            (FEATURES[0], "NaN", "numeric|finite"),
            (FEATURES[0], "1e100", "float32"),
            (TARGET, "OTHER", "Unknown Class"),
        )
        for column, value, message in cases:
            with self.subTest(value=value):
                invalid = source.copy()
                invalid.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    parse_csv(invalid.to_csv(index=False).encode())
        with self.assertRaisesRegex(ValueError, "at most"):
            parse_csv(source.to_csv(index=False).encode(), max_rows=1)
        missing = sample_data(2)
        missing.loc[0, FEATURES[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_data(missing)

    def test_download_reads_original_arff_and_refuses_source_overwrite(self):
        source = sample_data()
        arff_lines = ["@RELATION beans"]
        arff_lines.extend(f"@ATTRIBUTE {feature} NUMERIC" for feature in FEATURES)
        arff_lines.append("@ATTRIBUTE Class {" + ",".join(LABELS) + "}")
        arff_lines.append("@DATA")
        arff_lines.extend(source.to_csv(index=False, header=False).splitlines())
        payload = BytesIO()
        with ZipFile(payload, "w") as archive:
            archive.writestr("DryBeanDataset/Dry_Bean_Dataset.arff", "\n".join(arff_lines))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "source.csv"
            with patch("data.urlopen", return_value=BytesIO(payload.getvalue())):
                downloaded = download_dataset(destination)
            pd.testing.assert_frame_equal(downloaded, source, check_exact=False)
            self.assertTrue(destination.is_file())
            with self.assertRaises(FileExistsError):
                download_dataset(destination)

    def test_preparation_writes_reloadable_partitions_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            sample_data().to_csv(source, index=False)
            output = root / "prepared"
            report = prepare(source, output)
            training = parse_csv((output / "train_data.csv").read_bytes(), max_rows=None)
            testing = parse_csv((output / "test_data.csv").read_bytes())
            self.assertEqual(len(training), report["training_rows"])
            self.assertEqual(len(testing), report["test_rows"])
            self.assertTrue(training.merge(testing, on=list(FEATURES)).empty)
            before = (output / "split.json").read_bytes()
            with self.assertRaises(FileExistsError):
                prepare(source, output)
            self.assertEqual((output / "split.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
