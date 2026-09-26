# Detector de fallas de GPU — Parte A

Clasifica `normal`, `sobrecalentamiento`, `degradacion_memoria` y
`falla_alimentacion` a partir de ventanas de telemetría. Implementa solamente la
parte A del enunciado: contrato, features, entrenamiento y serialización.

## Reproducir en PowerShell

Ejecutar desde la raíz del repositorio. Entorno verificado: Python 3.13.7.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.train --exclude-invalid-windows
.venv\Scripts\python -m unittest discover -s tests -v
```

El CSV original es `data/telemetria_publica (1).csv` y se conserva intacto.
Se puede seleccionar otro con `--data "ruta.csv"` y el tamaño de ventana con
`--window-size 30` (mínimo 10, divisor de 300).

## Contrato y calidad de datos

`src/schema.py` valida con Pandera columnas, tipos, ausencia de nulos y valores
finitos: temperatura entre -40 y 150 °C, potencia en (0, 500] W, utilización
entre 0 y 100 %, reloj en (0, 5000] MHz y ECC entero no negativo. Son límites
amplios de plausibilidad, no umbrales para diagnosticar fallas. Exige etiquetas
válidas y pares episodio/segundo únicos. Además se verifica que cada episodio
original tenga exactamente los segundos 0..299 y una sola etiqueta.

El archivo contiene 14 400 filas, 48 episodios y 12 episodios por clase.
Hay tres potencias negativas: episodio/segundo 36/107, 38/262 y 44/158.
Al detectar potencia negativa, la validación emite una notificación de nivel
WARNING en consola con la cantidad de lecturas y hasta diez índices de ejemplo.
La notificación no sustituye el rechazo de esos datos por el contrato.
Por defecto, el entrenamiento aborta ante datos inválidos. La opción explícita
`--exclude-invalid-windows` registra las lecturas y errores en `reports/` y
excluye las tres ventanas completas afectadas (90 filas). No imputa valores ni
une segundos separados por huecos. El conjunto restante vuelve a validarse.

## Features y evaluación

`src/features.py` ordena por segundo y construye ventanas de 30 segundos dentro
de cada episodio. Cada ventana hereda la etiqueta de su episodio. Calcula 22
features: media, desviación muestral, mínimo y máximo de las cinco señales,
total de ECC y rango de potencia. La temperatura y el reloj ayudan a identificar
sobrecalentamiento; ECC revela degradación de memoria; la variabilidad de potencia
y reloj ayuda a detectar fallas de alimentación. No usa `estado`, `segundo` ni
`episodio_id` como entradas del clasificador.

Se reserva el 25 % de los episodios mediante partición estratificada con semilla
42: 36 episodios para entrenamiento y 12 para evaluación. No se comparten
episodios entre ambos conjuntos. Se usa Random Forest con 300 árboles; no requiere
escalado. La evaluación obtuvo accuracy 1.0 sobre los episodios reservados.
Este resultado corresponde a una partición del dataset público y no garantiza
el desempeño en las ventanas ocultas del profesor.

`reports/metrics.json` contiene los episodios de cada partición, conteos,
accuracy, métricas por clase, matriz de confusión y hash del CSV. Tras evaluar,
se reentrena sobre las 477 ventanas válidas para producir el artefacto final.

## Pipeline serializado

`models/modelo.joblib` contiene el pipeline completo: `WindowFeatures` y
`RandomForestClassifier`. La extracción de features viaja con el clasificador.
Se comprobó que las predicciones y probabilidades coinciden antes y después de
serializar. Para cargarlo se requieren las dependencias y el paquete `src`.

```python
import joblib
import pandas as pd
from src.schema import SIGNALS

modelo = joblib.load("models/modelo.joblib")
datos = pd.read_csv("data/telemetria_publica (1).csv")
ventana = datos[datos.episodio_id == 0].sort_values("segundo").iloc[:30]
lecturas = ventana[list(SIGNALS)].to_dict("records")
estado = modelo.predict([lecturas])[0]
confianza = modelo.predict_proba([lecturas]).max()
print(estado, confianza)
```

La entrada del pipeline es una lista de ventanas crudas. Solo se deben cargar
archivos joblib de confianza. La API y Docker corresponden a la parte B y no
están implementados.
