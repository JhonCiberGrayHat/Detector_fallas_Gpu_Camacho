"""Pruebas del contrato, las ventanas y el artefacto de la parte A."""

import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandera as pa

from src.features import extract_features, make_windows
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


if __name__ == "__main__":
    unittest.main()
