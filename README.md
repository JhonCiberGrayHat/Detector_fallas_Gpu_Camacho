# Detector de fallas de GPU

Clasifica `normal`, `sobrecalentamiento`, `degradacion_memoria` y
`falla_alimentacion` a partir de ventanas de telemetría. Implementa las
partes A y B del enunciado: modelo, validación, API y contenedor Docker.

## Estructura del proyecto

```text
Detector_fallas_Gpu_Camacho/
├── data/                        # Telemetría original utilizada para entrenar y evaluar.
│   └── telemetria_publica (1).csv
├── src/                         # Módulos de validación, features y servicio de inferencia.
│   ├── __init__.py               # Permite importar src como paquete de Python.
│   ├── schema.py                # Contrato de datos con Pandera.
│   ├── features.py              # Ventanas y características compartidas por modelo y API.
│   └── api.py                   # FastAPI, validación de peticiones y POST /predecir.
├── models/                      # Artefactos entrenados que carga la API.
│   └── modelo.joblib            # Pipeline completo: features y clasificador.
├── reports/                     # Evidencias de validación y evaluación del entrenamiento.
│   ├── metrics.json             # Métricas, particiones y hash del dataset.
│   ├── invalid_readings.csv     # Lecturas excluidas por incumplir el contrato.
│   └── validation_errors.csv    # Detalle de los errores detectados por Pandera.
├── tests/                       # Pruebas del entrenamiento, validación y API.
│   ├── test_part_a.py
│   └── test_api.py
├── train.py                     # Entrada de entrenamiento, situada en la raíz.
├── requirements.txt             # Dependencias de entrenamiento, API y pruebas.
├── Dockerfile                   # Construye la imagen y arranca Uvicorn.
├── .dockerignore                # Limita los archivos enviados a la construcción Docker.
├── .gitignore                   # Excluye archivos locales y generados de Git.
├── ENUNCIADO_PARCIAL.md          # Requisitos del proyecto.
└── README.md                    # Documentación e instrucciones de ejecución.
```

Al instalar localmente se crea `.venv/`, que contiene el entorno virtual y sus
dependencias. Python puede crear carpetas `__pycache__/` para almacenar código
compilado. Ambas se generan en cada equipo y no se incluyen en Git.

## Levantar el proyecto desde una clonación nueva

Los siguientes comandos se ejecutan en PowerShell. Se requiere Git y Docker
Desktop iniciado, con contenedores Linux y el puerto 8000 disponible. Para
entrenar o ejecutar las pruebas localmente, instalar también Python 3.12 o 3.13
(el entorno local se verificó con Python 3.13.7).

### 1. Clonar y entrar en el proyecto

```powershell
git clone https://github.com/JhonCiberGrayHat/Detector_fallas_Gpu_Camacho.git
cd Detector_fallas_Gpu_Camacho
```

Ejecutar los siguientes pasos desde esta carpeta raíz, donde está `train.py`.

### 2. Construir y levantar el servicio completo

El repositorio incluye `models/modelo.joblib`, por lo que se puede iniciar la
API directamente sin instalar Python en el equipo ni volver a entrenar:

```powershell
docker build -t detector-gpu .
docker run --rm --name detector-gpu-api -p 8000:8000 detector-gpu
```

Mantener esta terminal abierta mientras se utiliza el servicio. La primera
construcción requiere conexión a Internet para descargar la imagen base y las
dependencias. El contenedor carga el pipeline ya entrenado al iniciar.

### 3. Comprobar que responde

Abrir http://localhost:8000/docs. En `POST /predecir`, pulsar **Try it out**,
introducir una ventana de al menos 10 lecturas y pulsar **Execute**. También se
puede ejecutar la [petición de ejemplo](#petición-de-ejemplo) en otra terminal.
Una petición válida devuelve HTTP 200 con `estado_predicho` y `confianza`.

Para detener el servicio, pulsar `Ctrl+C` en su terminal o ejecutar desde otra:

```powershell
docker stop detector-gpu-api
```

### 4. Reproducir el entrenamiento y las pruebas (opcional)

Ejecutar desde la raíz del repositorio. Entorno verificado: Python 3.13.7.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python train.py --exclude-invalid-windows
.venv\Scripts\python -m unittest discover -s tests -v
```

No es necesario activar el entorno virtual: los comandos usan directamente su
intérprete. El entrenamiento actualiza `models/modelo.joblib` y `reports/`.
Después de reentrenar, detener el contenedor anterior y repetir el paso 2 para
que la imagen incorpore el nuevo modelo.

El CSV original es `data/telemetria_publica (1).csv` y se conserva intacto.
Se puede seleccionar otro con `--data "ruta.csv"` y el tamaño de ventana con
`--window-size 30` (mínimo 10, divisor de 300).

## Contrato y calidad de datos

`src/schema.py` valida con Pandera columnas, tipos, ausencia de nulos y valores
finitos: temperatura entre -40 y 150 °C, potencia en (0, 500] W, utilización
entre 0 y 100 %, reloj en (500, 4000] MHz y ECC entero no negativo. Son límites
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
se guarda el mismo pipeline ajustado con las 358 ventanas de entrenamiento.
Las 119 ventanas de test se utilizan exclusivamente para evaluar: nunca se pasan
a `fit` ni a `fit_transform`, tampoco después de calcular las métricas.

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
archivos joblib de confianza.

## Parte B: ejecutar con Docker

Desde la raíz, con Docker iniciado y configurado para contenedores Linux:

```powershell
docker build -t detector-gpu .
docker run -p 8000:8000 detector-gpu
```

Abrir http://localhost:8000/docs para consultar y probar `POST /predecir`.
La imagen usa `python:3.12-slim`, instala las dependencias y copia `src/` y el
modelo ya entrenado. No requiere el CSV ni entrena al arrancar. `.dockerignore`
mantiene `train.py` (ubicado en la raíz) fuera de la imagen de inferencia y
excluye el entorno local, Git, datos y reportes del contexto de construcción.
Si falta el modelo, el servicio falla al iniciar en vez de entrenar otro.

Para ejecutar localmente sin Docker:

```powershell
.venv\Scripts\python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

## Petición de ejemplo

Con el servicio iniciado, ejecutar en otra terminal PowerShell:

```powershell
$lecturas = 1..30 | ForEach-Object {
    @{temp_c=89.8; power_w=298.9; util_pct=95.4; clock_mhz=2091; ecc_errors=0}
}
$body = @{lecturas=$lecturas} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Uri http://localhost:8000/predecir -Method Post -ContentType 'application/json' -Body $body
```

Devuelve `estado_predicho` y `confianza` (probabilidad de la clase seleccionada
según el modelo, no una garantía de acierto). El endpoint ejecuta el pipeline
guardado, incluida la misma extracción de features que usa el entrenamiento.

La petición debe contener `lecturas`, con al menos 10 lecturas consecutivas de
una misma GPU, una por segundo. Se recomiendan 30, como en el entrenamiento.
No se envían `estado`, `episodio_id` ni `segundo`. Sin esos identificadores, el
cliente es responsable de mantener el orden y la procedencia de las lecturas.

Pydantic valida los cinco campos obligatorios, tipos estrictos, valores finitos
y los mismos rangos físicos del contrato de entrenamiento. Rechaza campos extra,
cadenas en lugar de números, ECC fraccionario, potencias fuera de (0, 500] y
ventanas con menos de 10 lecturas. La respuesta es HTTP 422 con `detail`, que
identifica el campo y la causa, sin ejecutar el modelo. La potencia negativa
también genera una advertencia en el registro del servidor. El ejemplo de dos
lecturas del enunciado ilustra el formato, pero no cumple el mínimo exigido.

## Pruebas

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

Incluyen comparación de la API con el pipeline para los cuatro estados,
rechazo de peticiones antes de la inferencia, documentación HTTP y fallo de
arranque si falta el modelo, además de las comprobaciones de la parte A.
