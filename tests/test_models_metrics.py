import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import matthews_corrcoef

from data import FEATURES, LABELS, TARGET
from metrics import METRIC_NAMES, comparison_table, evaluate
from models import MODEL_FILES, compare_training_folds, make_models
from test_data import sample_data


class ModelTests(unittest.TestCase):
    def test_estimators_are_fresh_and_scaling_is_inside_svm_calibration(self):
        first, second = make_models(), make_models()
        self.assertEqual(list(first), list(MODEL_FILES))
        for name in MODEL_FILES:
            self.assertIsNot(first[name], second[name])
            with self.assertRaises(NotFittedError):
                first[name].predict(sample_data(1).loc[:, list(FEATURES)])
        calibrator = first["Support Vector Machine"].named_steps["classifier"]
        self.assertEqual(list(calibrator.estimator.named_steps), ["scale", "svm"])
        self.assertNotIn("scale", first["Support Vector Machine"].named_steps)
        self.assertFalse(calibrator.ensemble)

    def test_every_pipeline_fits_and_evaluates_unseen_synthetic_rows(self):
        source = sample_data(20)
        training = source.groupby(TARGET, sort=False).head(15)
        testing = source.drop(training.index)
        features = training.loc[:, list(FEATURES)]
        for name, model in make_models().items():
            with self.subTest(model=name):
                model.fit(features, training[TARGET])
                result = evaluate(model, testing)
                self.assertEqual(tuple(result["metrics"]), METRIC_NAMES)
                self.assertTrue(all(np.isfinite(value) for value in result["metrics"].values()))
                self.assertTrue(all(0 <= result["metrics"][key] <= 1 for key in METRIC_NAMES[:-1]))
                self.assertTrue(-1 <= result["metrics"]["MCC"] <= 1)
                self.assertEqual(np.asarray(result["confusion_matrix"]).sum(), len(testing))
                if "scale" in model.named_steps:
                    np.testing.assert_allclose(model.named_steps["scale"].mean_, features.mean())
                if name == "Support Vector Machine":
                    fitted = model.named_steps["classifier"].calibrated_classifiers_[0].estimator
                    np.testing.assert_allclose(fitted.named_steps["scale"].mean_, features.mean())

    def test_cross_validation_reuses_identical_folds_and_only_training_rows(self):
        training = sample_data(6)
        scores = {
            "train_accuracy": np.array([0.9, 0.9, 0.9]),
            "test_accuracy": np.array([0.8, 0.8, 0.8]),
            "test_f1_macro": np.array([0.7, 0.8, 0.9]),
        }
        with patch("models.cross_validate", return_value=scores) as runner:
            result = compare_training_folds(training)
        self.assertEqual(runner.call_count, 5)
        calls = runner.call_args_list
        for call in calls:
            self.assertIs(call.kwargs["cv"], calls[0].kwargs["cv"])
            self.assertTrue(call.args[1].equals(training.loc[:, list(FEATURES)]))
        self.assertAlmostEqual(result["models"]["KNN"]["validation_macro_f1"], 0.8)
        incomplete = training[training[TARGET] != LABELS[-1]]
        for invalid in (sample_data(5), incomplete):
            with self.assertRaisesRegex(ValueError, "all seven classes and six"):
                compare_training_folds(invalid)

    def test_real_nested_cross_validation_produces_five_model_results(self):
        result = compare_training_folds(sample_data(6))
        self.assertEqual(set(result["models"]), set(MODEL_FILES))
        self.assertIn(result["selected_model"], MODEL_FILES)
        for model in result["models"].values():
            self.assertEqual(len(model["fold_macro_f1"]), 3)
            self.assertTrue(np.isfinite(model["validation_macro_f1"]))


class MetricTests(unittest.TestCase):
    def fixture(self):
        frame = sample_data(2)
        classes = list(reversed(LABELS))
        model = Mock()
        model.classes_ = np.array(classes)
        model.predict.return_value = frame[TARGET].to_numpy()
        model.predict_proba.return_value = np.array([
            [0.94 if label == target else 0.01 for label in classes]
            for target in frame[TARGET]
        ])
        return frame, model

    def test_class_probability_alignment_and_six_perfect_scores(self):
        frame, model = self.fixture()
        result = evaluate(model, frame)
        self.assertEqual(tuple(result["metrics"]), METRIC_NAMES)
        for score in result["metrics"].values():
            self.assertAlmostEqual(score, 1)
        self.assertIsNone(result["auc_explanation"])
        self.assertEqual(np.asarray(result["confusion_matrix"]).shape, (7, 7))
        table = comparison_table({"example": result})
        self.assertEqual(table.index.name, "Model")
        self.assertEqual(table.shape, (1, 6))

    def test_nonperfect_scores_use_macro_averages_and_multiclass_mcc(self):
        frame, model = self.fixture()
        predicted = frame[TARGET].to_numpy().copy()
        predicted[0] = LABELS[1]
        model.predict.return_value = predicted
        result = evaluate(model, frame)
        scores = result["metrics"]
        self.assertAlmostEqual(scores["Accuracy"], 13 / 14)
        self.assertAlmostEqual(scores["Precision"], (6 + 2 / 3) / 7)
        self.assertAlmostEqual(scores["Recall"], 6.5 / 7)
        self.assertAlmostEqual(scores["F1"], (5 + 2 / 3 + 0.8) / 7)
        self.assertAlmostEqual(scores["MCC"], matthews_corrcoef(frame[TARGET], predicted))
        self.assertEqual(result["confusion_matrix"][0][1], 1)

    def test_missing_classes_keep_fixed_reports_and_explain_undefined_auc(self):
        frame, model = self.fixture()
        for count in (1, 4):
            with self.subTest(rows=count):
                model.predict.return_value = frame[TARGET].to_numpy()[:count]
                model.predict_proba.return_value = self.fixture()[1].predict_proba.return_value[:count]
                result = evaluate(model, frame.iloc[:count])
                self.assertIsNone(result["metrics"]["AUC"])
                self.assertIn("requires all seven", result["auc_explanation"])
                self.assertEqual(np.asarray(result["confusion_matrix"]).shape, (7, 7))
                self.assertTrue(set(LABELS).issubset(result["classification_report"]))
                if count == 1:
                    self.assertEqual(result["metrics"]["MCC"], 0)
                    self.assertAlmostEqual(result["metrics"]["F1"], 1 / 7)

    def test_invalid_predictions_probabilities_and_classes_are_rejected(self):
        for probabilities in (
            np.ones((14, 7)), np.full((14, 7), np.nan),
            np.ones((14, 6)), np.full((14, 7), -0.1),
            np.full((14, 7), np.inf),
        ):
            frame, model = self.fixture()
            model.predict_proba.return_value = probabilities
            with self.assertRaisesRegex(ValueError, "probabilit"):
                evaluate(model, frame)
        for predictions in (np.repeat("OTHER", 14), np.array([LABELS[0]]), np.full((14, 1), LABELS[0])):
            frame, model = self.fixture()
            model.predict.return_value = predictions
            with self.assertRaisesRegex(ValueError, "invalid predictions"):
                evaluate(model, frame)
        frame, model = self.fixture()
        model.classes_ = np.array(LABELS[:-1])
        with self.assertRaisesRegex(ValueError, "all seven"):
            evaluate(model, frame)


if __name__ == "__main__":
    unittest.main()
