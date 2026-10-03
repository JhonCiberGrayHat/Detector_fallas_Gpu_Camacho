FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Solo módulos de inferencia; train.py se ejecuta desde la raíz antes de construir.
COPY src/__init__.py src/api.py src/features.py src/schema.py ./src/
COPY models/modelo.joblib ./models/modelo.joblib

EXPOSE 8000
##CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}