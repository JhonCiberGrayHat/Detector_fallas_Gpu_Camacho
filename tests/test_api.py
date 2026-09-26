"""Pruebas HTTP con el pipeline real y validación previa a la inferencia."""

import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from src.api import app, MODEL_PATH
from src.schema import SIGNALS


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(MODEL_PATH.parents[1] / "data/telemetria_publica (1).csv")

    def payload(self, episode=0, length=30):
        window = self.data[self.data.episodio_id == episode].sort_values("segundo").iloc[:length]
        return {"lecturas": window[list(SIGNALS)].to_dict("records")}

    def test_predictions_match_pipeline_for_all_states(self):
        with TestClient(app) as client:
            for episode in (0, 12, 24, 36):
                payload = self.payload(episode)
                expected = app.state.modelo.predict([payload["lecturas"]])[0]
                confidence = app.state.modelo.predict_proba([payload["lecturas"]]).max()
                response = client.post("/predecir", json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["estado_predicho"], expected)
                self.assertAlmostEqual(response.json()["confianza"], confidence)

    def test_invalid_requests_never_reach_model(self):
        cases = [{}, {"lecturas": []}, self.payload(length=3), self.payload(length=9)]
        for field, value in [("power_w", -1), ("power_w", 500.1), ("power_w", 0),
                             ("temp_c", 151), ("util_pct", 101), ("clock_mhz", 500),
                             ("clock_mhz", 4001), ("ecc_errors", -1), ("ecc_errors", 0.5),
                             ("ecc_errors", True), ("temp_c", "88.3"), ("temp_c", None)]:
            payload = self.payload()
            payload["lecturas"][0][field] = value
            cases.append(payload)
        missing = self.payload()
        del missing["lecturas"][0]["power_w"]
        cases.append(missing)
        extra = self.payload()
        extra["lecturas"][0]["estado"] = "normal"
        cases.append(extra)
        with TestClient(app) as client, \
                patch.object(app.state.modelo, "predict_proba") as predict, \
                patch.object(app.state.modelo, "fit") as fit:
            for payload in cases:
                with self.subTest(payload=payload):
                    response = client.post("/predecir", json=payload)
                    self.assertEqual(response.status_code, 422)
                    self.assertIn("detail", response.json())
            predict.assert_not_called()
            fit.assert_not_called()

    def test_docs_and_minimum_window(self):
        with TestClient(app) as client:
            self.assertEqual(client.get("/docs").status_code, 200)
            self.assertEqual(client.get("/openapi.json").status_code, 200)
            self.assertEqual(client.post("/predecir", json=self.payload(length=10)).status_code, 200)

    def test_nonfinite_and_malformed_json_rejected(self):
        import json
        with TestClient(app) as client, patch.object(app.state.modelo, "predict_proba") as predict:
            for value in (float("inf"), float("nan")):
                payload = self.payload()
                payload["lecturas"][0]["temp_c"] = value
                response = client.post("/predecir", content=json.dumps(payload),
                                       headers={"Content-Type": "application/json"})
                self.assertEqual(response.status_code, 422)
            self.assertEqual(client.post("/predecir", content="{",
                             headers={"Content-Type": "application/json"}).status_code, 422)
            predict.assert_not_called()

    def test_missing_model_fails_at_startup(self):
        with patch("src.api.MODEL_PATH", MODEL_PATH.with_name("missing.joblib")):
            with self.assertRaises(FileNotFoundError), TestClient(app):
                pass


if __name__ == "__main__":
    unittest.main()
