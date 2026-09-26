"""API de inferencia: carga el pipeline entrenado una vez al iniciar."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

# train.py, en la raíz, guarda el pipeline aquí; la API no importa el entrenador.
MODEL_PATH = Path(__file__).resolve().parents[1] / "models/modelo.joblib"
logger = logging.getLogger(__name__)


class Lectura(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    temp_c: float = Field(ge=-40, le=150)
    power_w: float = Field(gt=0, le=500)
    util_pct: float = Field(ge=0, le=100)
    clock_mhz: float = Field(gt=500, le=4000)
    ecc_errors: int = Field(ge=0)

    @field_validator("power_w", mode="before")
    @classmethod
    def notify_negative_power(cls, value):
        if isinstance(value, (int, float)) and value < 0:
            logger.warning("Potencia negativa detectada en petición: power_w=%s", value)
        return value


class Ventana(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    lecturas: list[Lectura] = Field(min_length=10, description="Una lectura por segundo; mínimo 10.")


class Prediccion(BaseModel):
    estado_predicho: Literal["normal", "sobrecalentamiento", "degradacion_memoria", "falla_alimentacion"]
    confianza: float = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.modelo = joblib.load(MODEL_PATH)
    yield
    del app.state.modelo


app = FastAPI(title="Detector de fallas de GPU", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError):
    # Evitar devolver valores no finitos del input, que no son serializables a JSON.
    details = [{key: item[key] for key in ("type", "loc", "msg")} for item in error.errors()]
    return JSONResponse(status_code=422, content={"detail": details})


@app.post("/predecir", response_model=Prediccion)
def predecir(ventana: Ventana, request: Request):
    # WindowFeatures, dentro del pipeline, reutiliza src/features.py sin hacer fit.
    lecturas = [lectura.model_dump() for lectura in ventana.lecturas]
    modelo = request.app.state.modelo
    probabilities = modelo.predict_proba([lecturas])[0]
    index = int(np.argmax(probabilities))
    return Prediccion(estado_predicho=str(modelo.classes_[index]),
                      confianza=float(probabilities[index]))
