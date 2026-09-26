"""Características compartidas para entrenamiento e inferencia."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.schema import SIGNALS


def extract_features(window): 
    """Recibe una ventana (DataFrame o lista de lecturas) sin usar etiquetas."""
    frame = pd.DataFrame(window) 
    if len(frame) < 10:
        raise ValueError("Se requieren al menos 10 lecturas por ventana.")
    values = frame.loc[:, list(SIGNALS)].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("La ventana contiene valores no finitos.")
    result = {}
    for signal in SIGNALS:
        series = values[signal]
        result.update({
            f"{signal}_mean": series.mean(),
            f"{signal}_std": series.std(ddof=1),
            f"{signal}_min": series.min(),
            f"{signal}_max": series.max(),
        })
    result["ecc_errors_total"] = values.ecc_errors.sum()
    result["power_w_range"] = values.power_w.max() - values.power_w.min()
    return result


class WindowFeatures(TransformerMixin, BaseEstimator):
    """Transformador serializable: una fila de features por ventana cruda."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame([extract_features(window) for window in X])


def make_windows(data, window_size=30):
    """Agrupa por episodio y tiempo original; rechaza ventanas incompletas."""
    if window_size < 10 or 300 % window_size:
        raise ValueError("window_size debe ser >= 10 y divisor de 300.")
    windows, labels, groups = [], [], []
    for (episode_id, _), window in data.groupby(
        [data.episodio_id, data.segundo // window_size], sort=True
    ):
        window = window.sort_values("segundo")
        if len(window) != window_size or not np.all(np.diff(window.segundo) == 1):
            raise ValueError("Ventana incompleta o no consecutiva.")
        if window.estado.nunique() != 1:
            raise ValueError("Una ventana debe tener una sola etiqueta.")
        windows.append(window.loc[:, list(SIGNALS)].reset_index(drop=True))
        labels.append(window.estado.iloc[0])
        groups.append(int(episode_id))
    return windows, np.asarray(labels), np.asarray(groups)
