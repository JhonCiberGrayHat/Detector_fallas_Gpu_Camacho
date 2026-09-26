"""Ejecutar desde la raíz: python train.py --exclude-invalid-windows."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandera as pa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features import WindowFeatures, make_windows
from src.schema import ESTADOS, validate_episodes, validate_telemetry

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/telemetria_publica (1).csv")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--exclude-invalid-windows", action="store_true",
                        help="Registra errores y excluye sus ventanas; por defecto aborta.")
    args = parser.parse_args()
    if args.window_size < 10 or 300 % args.window_size:
        parser.error("El tamaño debe ser >=10 y divisor de 300.")
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    data = pd.read_csv(args.data)
    original_rows = len(data)
    excluded = []
    try:
        validate_telemetry(data)
    except pa.errors.SchemaErrors as error:
        error.failure_cases.to_csv(reports / "validation_errors.csv", index=False)
        if not args.exclude_invalid_windows:
            raise
        failures = error.failure_cases
        # Errores estructurales o de tipos no se corrigen automáticamente.
        if failures["index"].isna().any():
            raise
        invalid_indices = failures["index"].astype(int).unique()
        invalid = data.loc[invalid_indices]
        invalid.to_csv(reports / "invalid_readings.csv", index=False)
        validate_episodes(data)
        bad_keys = set(zip(invalid.episodio_id, invalid.segundo // args.window_size))
        excluded = [{"episodio_id": int(e), "ventana": int(w)} for e, w in sorted(bad_keys)]
        keep = [(e, s // args.window_size) not in bad_keys
                for e, s in zip(data.episodio_id, data.segundo)]
        data = data.loc[keep].copy()
        validate_telemetry(data)
    else:
        validate_episodes(data)
    windows, labels, groups = make_windows(data, args.window_size)
    episodes = data.groupby("episodio_id").estado.first()
    train_ids, test_ids = train_test_split(
        episodes.index.to_numpy(), test_size=0.25, random_state=42, stratify=episodes.values
    )
    train_idx = np.flatnonzero(np.isin(groups, train_ids))
    test_idx = np.flatnonzero(np.isin(groups, test_ids))
    if set(train_ids) & set(test_ids) or np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Entrenamiento y test deben ser disjuntos por episodio y ventana.")
    train_windows = [windows[i] for i in train_idx]
    test_windows = [windows[i] for i in test_idx]
    pipeline = Pipeline([
        ("features", WindowFeatures()),
        ("classifier", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                               random_state=42, n_jobs=-1)),
    ])
    # Único ajuste: todos los pasos del pipeline reciben exclusivamente entrenamiento.
    pipeline.fit(train_windows, labels[train_idx])
    predicted = pipeline.predict(test_windows)
    report = {
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "seed": 42, "window_size": args.window_size,
        "original_rows": original_rows, "accepted_rows": len(data),
        "excluded_windows": excluded, "total_windows": len(windows),
        "train_episodes": sorted(train_ids.tolist()), "test_episodes": sorted(test_ids.tolist()),
        "train_windows": len(train_idx), "test_windows": len(test_idx),
        "accuracy": accuracy_score(labels[test_idx], predicted),
        "classification_report": classification_report(labels[test_idx], predicted, output_dict=True),
        "confusion_matrix_labels": list(ESTADOS),
        "confusion_matrix": confusion_matrix(labels[test_idx], predicted, labels=ESTADOS).tolist(),
        "serialized_model": "Ajustado exclusivamente con entrenamiento; test solo para evaluación.",
    }
    # Guardar exactamente el modelo evaluado, sin volver a ajustar sobre test.
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    model_path = models / "modelo.joblib"
    joblib.dump(pipeline, model_path)
    restored = joblib.load(model_path)
    np.testing.assert_array_equal(pipeline.predict(windows), restored.predict(windows))
    np.testing.assert_allclose(pipeline.predict_proba(windows), restored.predict_proba(windows))
    (reports / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Ventanas válidas: {len(windows)}; excluidas: {len(excluded)}")
    print(f"Accuracy en episodios reservados: {report['accuracy']:.4f}")
    print(f"Pipeline completo guardado y recargado: {model_path}")


if __name__ == "__main__":
    main()
