"""Contrato de telemetría previo a la extracción de características."""

import logging

import numpy as np
import pandas as pd
import pandera.pandas as pa

ESTADOS = ("normal", "sobrecalentamiento", "degradacion_memoria", "falla_alimentacion")
SIGNALS = ("temp_c", "power_w", "util_pct", "clock_mhz", "ecc_errors")
logger = logging.getLogger(__name__)

def numeric(dtype, checks):
    return pa.Column(dtype, [*checks, pa.Check(np.isfinite)], nullable=False)


TELEMETRIA_SCHEMA = pa.DataFrameSchema(
    {
        "episodio_id": numeric(int, [pa.Check.ge(0)]), ## >= que cero
        "segundo": numeric(int, [pa.Check.in_range(0, 299)]), 
        "temp_c": numeric(float, [pa.Check.in_range(-40, 150)]),
        "power_w": numeric(float, [pa.Check.gt(0), pa.Check.le(500)]),
        "util_pct": numeric(float, [pa.Check.in_range(0, 100)]),
        "clock_mhz": numeric(float, [pa.Check.gt(500), pa.Check.le(4000)]),
        "ecc_errors": numeric(int, [pa.Check.ge(0)]),
        "estado": pa.Column(str, pa.Check.isin(ESTADOS)),
    },
    strict=True,
    unique=["episodio_id", "segundo"], ## para que se verifique que no se repitan los segundos (valores repetidos)
)


def validate_episodes(data):
    """Comprueba estructura original sin aceptar huecos ni etiquetas mixtas."""
    for episode_id, episode in data.groupby("episodio_id", dropna=False):
        if (len(episode) != 300
                or sorted(episode.segundo.tolist()) != list(range(300))
                or episode.estado.nunique(dropna=False) != 1):
            raise ValueError(f"Episodio {episode_id}: se requieren segundos 0..299 y un solo estado.")


def validate_telemetry(data): ## para validar y notificar los datos de potencia negativos
    if "power_w" in data.columns:
        negative = pd.to_numeric(data["power_w"], errors="coerce").lt(0)
        if negative.any():
            logger.warning(
                "Potencia negativa detectada: %d lectura(s) con power_w < 0; "
                "índices de ejemplo: %s. Estos datos no cumplen el contrato.",
                int(negative.sum()), data.index[negative].tolist()[:10],
            )
    return TELEMETRIA_SCHEMA.validate(data, lazy=True)
