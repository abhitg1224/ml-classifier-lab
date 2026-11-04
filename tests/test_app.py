import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from data import LABELS, parse_csv
from metrics import METRIC_NAMES, comparison_table
from models import MODEL_FILES


ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()

    def test_actual_upload_and_all_model_selections_are_inference_only(self):
        expected = json.loads((ROOT / "results.json").read_text())["models"]
        with patch.object(Pipeline, "fit", side_effect=AssertionError("App must not fit")):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric), 0)
            self.assertEqual(len(app.info), 1)
            app.file_uploader(key="test_csv").set_value(
                ("test_data.csv", (ROOT / "test_data.csv").read_bytes(), "text/csv")
            ).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            pd.testing.assert_frame_equal(
                app.dataframe[0].value, comparison_table(expected), check_names=False,
            )
            for name in MODEL_FILES:
                app.selectbox(key="classifier").select(name).run()
                self.assertFalse(app.exception)
                self.assertEqual([item.label for item in app.metric], list(METRIC_NAMES))
                self.assertEqual(
                    [item.value for item in app.metric],
                    [f'{expected[name]["metrics"][metric]:.6f}' for metric in METRIC_NAMES],
                )
                self.assertEqual(
                    app.dataframe[1].value.to_numpy().tolist(), expected[name]["confusion_matrix"],
                )
                self.assertEqual(app.dataframe[1].value.shape, (7, 7))
                self.assertEqual(list(app.dataframe[2].value.index[:7]), list(LABELS))

    def test_invalid_upload_overrides_bundled_data_and_clears_previous_results(self):
        with patch.object(Pipeline, "fit", side_effect=AssertionError("App must not fit")):
            app = self.app()
            app.checkbox(key="bundled").check().run()
            self.assertEqual(len(app.metric), 6)
            with self.assertLogs("classifier_app", level="ERROR"):
                app.file_uploader(key="test_csv").set_value(
                    ("invalid.csv", b"wrong,Class\n1,SIRA\n", "text/csv")
                ).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.error), 1)
            self.assertIn("Incorrect columns", app.error[0].value)
            self.assertEqual(len(app.metric), 0)
            self.assertEqual(len(app.dataframe), 0)
            app.file_uploader(key="test_csv").clear().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.metric), 6)
            app.checkbox(key="bundled").uncheck().run()
            self.assertEqual(len(app.metric), 0)
            self.assertEqual(len(app.dataframe), 0)

    def test_partial_class_upload_explains_auc_and_keeps_fixed_reports(self):
        frame = parse_csv((ROOT / "test_data.csv").read_bytes()).iloc[:1]
        with patch.object(Pipeline, "fit", side_effect=AssertionError("App must not fit")):
            app = self.app()
            app.file_uploader(key="test_csv").set_value(
                ("one_row.csv", frame.to_csv(index=False).encode(), "text/csv")
            ).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.metric[1].value, "N/A")
            self.assertEqual(app.metric[5].value, "0.000000")
            self.assertIn("requires all seven classes", app.warning[0].value)
            self.assertTrue(app.dataframe[0].value["AUC"].isna().all())
            self.assertEqual(app.dataframe[1].value.shape, (7, 7))
            self.assertEqual(app.dataframe[1].value.to_numpy().sum(), 1)


if __name__ == "__main__":
    unittest.main()
