"""Pruebas del contrato, las ventanas y el artefacto de la parte A."""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pandera as pa

from src.features import WindowFeatures, extract_features, make_windows
import train
from src.schema import SIGNALS, validate_episodes, validate_telemetry

ROOT = Path(__file__).resolve().parents[1]


class PartATests(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(ROOT / "data/telemetria_publica (1).csv")
        self.episode = self.data[self.data.episodio_id == 0].copy()

    def test_contract_rejects_corruption(self):
        validate_telemetry(self.episode)
        for column, value in [("temp_c", 151.0), ("power_w", -1.0),
                              ("ecc_errors", -1), ("estado", "desconocido"),
                              ("util_pct", np.nan), ("clock_mhz", np.inf)]:
            with self.subTest(column=column):
                broken = self.episode.copy()
                broken.loc[0, column] = value
                with self.assertRaises(pa.errors.SchemaErrors):
                    validate_telemetry(broken)

    def test_windows_do_not_mix_episodes(self):
        data = self.data[self.data.episodio_id.isin([0, 12])]
        windows, labels, groups = make_windows(data.sample(frac=1, random_state=42))
        self.assertEqual(len(windows), 20)
        for window, label, group in zip(windows, labels, groups):
            self.assertEqual(len(window), 30)
            self.assertEqual(list(window.columns), list(SIGNALS))
            self.assertEqual(label, data[data.episodio_id == group].estado.iloc[0])
            source = data[data.episodio_id == group]
            self.assertTrue(window.temp_c.isin(source.temp_c).all())

    def test_power_limit(self):
        self.episode.loc[0, "power_w"] = 500.0
        with self.assertNoLogs("src.schema", level="WARNING"):
            validate_telemetry(self.episode)
        for value in (500.1, 0.0):
            self.episode.loc[0, "power_w"] = value
            with self.assertRaises(pa.errors.SchemaErrors):
                validate_telemetry(self.episode)

    def test_negative_power_notifies_and_rejects(self):
        self.episode.loc[[0, 1], "power_w"] = -1.0
        with self.assertLogs("src.schema", level="WARNING") as notification:
            with self.assertRaises(pa.errors.SchemaErrors):
                validate_telemetry(self.episode)
        self.assertIn("Potencia negativa detectada: 2 lectura(s)", notification.output[0])
        self.assertIn("[0, 1]", notification.output[0])

    def test_missing_seconds_rejected(self):
        with self.assertRaises(ValueError):
            validate_episodes(self.episode.iloc[1:])
        with self.assertRaises(ValueError):
            make_windows(self.episode.iloc[1:])

    def test_features_ignore_identifiers_and_label(self):
        window = self.episode.iloc[:30].copy()
        expected = extract_features(window)
        window["estado"] = "otra"
        window["episodio_id"] = 999
        window["segundo"] = -1
        self.assertEqual(expected, extract_features(window))
        with self.assertRaises(ValueError):
            extract_features(window.iloc[:3])

    def test_saved_pipeline_accepts_raw_readings(self):
        pipeline = joblib.load(ROOT / "models/modelo.joblib")
        readings = self.episode.iloc[:30][list(SIGNALS)].to_dict("records")
        self.assertEqual(pipeline.predict([readings]).tolist(), ["normal"])
        self.assertAlmostEqual(pipeline.predict_proba([readings]).sum(), 1.0)

    def test_training_never_fits_test_data(self):
        feature_inputs, classifier_inputs = [], []
        original_features_fit = WindowFeatures.fit
        original_classifier_fit = train.RandomForestClassifier.fit

        def record_features(instance, X, y=None):
            feature_inputs.append(X)
            return original_features_fit(instance, X, y)

        def record_classifier(instance, X, y, **kwargs):
            classifier_inputs.append((X.copy(), y.copy()))
            return original_classifier_fit(instance, X, y, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            argv = ["train", "--data", str(ROOT / "data/telemetria_publica (1).csv"),
                    "--exclude-invalid-windows"]
            with patch.object(train, "ROOT", output), patch("sys.argv", argv), \
                    patch.object(WindowFeatures, "fit", record_features), \
                    patch.object(train.RandomForestClassifier, "fit", record_classifier):
                train.main()
            report = json.loads((output / "reports/metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(len(feature_inputs), 1)
        self.assertEqual(len(classifier_inputs), 1)
        self.assertTrue(set(report["train_episodes"]).isdisjoint(report["test_episodes"]))
        clean = self.data.copy()
        for excluded in report["excluded_windows"]:
            clean = clean.loc[~((clean.episodio_id == excluded["episodio_id"])
                               & (clean.segundo // 30 == excluded["ventana"]))]
        train_data = clean[clean.episodio_id.isin(report["train_episodes"])]
        expected_windows, expected_labels, _ = make_windows(train_data)
        self.assertEqual(len(feature_inputs[0]), len(expected_windows))
        for actual, expected in zip(feature_inputs[0], expected_windows):
            pd.testing.assert_frame_equal(actual, expected)
        pd.testing.assert_frame_equal(classifier_inputs[0][0],
                                      WindowFeatures().transform(expected_windows))
        np.testing.assert_array_equal(classifier_inputs[0][1], expected_labels)

    def test_features_have_expected_statistics(self):
        window = pd.DataFrame({signal: np.arange(1, 11, dtype=float) for signal in SIGNALS})
        features = extract_features(window)
        self.assertEqual(len(features), 22)
        for signal in SIGNALS:
            self.assertEqual(features[f"{signal}_mean"], 5.5)
            self.assertAlmostEqual(features[f"{signal}_std"], (55 / 6) ** 0.5)
            self.assertEqual(features[f"{signal}_min"], 1)
            self.assertEqual(features[f"{signal}_max"], 10)
        self.assertEqual(features["ecc_errors_total"], 55)
        self.assertEqual(features["power_w_range"], 9)

    def test_structure_and_types_rejected(self):
        fractional_ecc = self.episode.astype({"ecc_errors": float})
        fractional_ecc.loc[0, "ecc_errors"] = 0.5
        cases = [self.episode.drop(columns="power_w"),
                 self.episode.assign(extra=1), fractional_ecc,
                 pd.concat([self.episode, self.episode.iloc[:1]], ignore_index=True)]
        for broken in cases:
            with self.subTest(columns=broken.columns.tolist(), rows=len(broken)):
                with self.assertRaises(pa.errors.SchemaErrors):
                    validate_telemetry(broken)
        self.episode.loc[0, "estado"] = "sobrecalentamiento"
        with self.assertRaises(ValueError):
            validate_episodes(self.episode)
        with self.assertRaises(ValueError):
            make_windows(self.episode)

    def test_invalid_data_aborts_before_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            argv = ["train", "--data", str(ROOT / "data/telemetria_publica (1).csv")]
            with patch.object(train, "ROOT", output), patch("sys.argv", argv), \
                    patch.object(train.Pipeline, "fit") as fit:
                with self.assertRaises(pa.errors.SchemaErrors):
                    train.main()
                fit.assert_not_called()
            self.assertTrue((output / "reports/validation_errors.csv").exists())
            self.assertFalse((output / "models/modelo.joblib").exists())

    def test_saved_model_matches_report_on_test(self):
        report = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(train.hashlib.sha256(
            (ROOT / "data/telemetria_publica (1).csv").read_bytes()).hexdigest(),
            report["dataset_sha256"])
        data = self.data[self.data.episodio_id.isin(report["test_episodes"])].copy()
        for excluded in report["excluded_windows"]:
            data = data.loc[~((data.episodio_id == excluded["episodio_id"])
                             & (data.segundo // report["window_size"] == excluded["ventana"]))]
        validate_telemetry(data)
        windows, labels, _ = make_windows(data, report["window_size"])
        pipeline = joblib.load(ROOT / "models/modelo.joblib")
        self.assertIsInstance(pipeline.named_steps["features"], WindowFeatures)
        self.assertEqual(set(pipeline.classes_), set(train.ESTADOS))
        predicted = pipeline.predict(windows)
        self.assertEqual(len(windows), report["test_windows"])
        self.assertEqual(train.accuracy_score(labels, predicted), report["accuracy"])
        self.assertEqual(train.confusion_matrix(labels, predicted,
                         labels=report["confusion_matrix_labels"]).tolist(),
                         report["confusion_matrix"])


if __name__ == "__main__":
    unittest.main()
