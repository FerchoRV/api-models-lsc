# =============================================================================
# Imagen para Colsign API (FastAPI + TensorFlow + MediaPipe + OpenCV).
#
# Diseñada para Google Cloud Run:
#   - Single stage, slim, ~1.5 GiB final (TF + MediaPipe son la mayor parte).
#   - Escucha en 0.0.0.0:$PORT (Cloud Run inyecta PORT=8080 por defecto).
#   - 1 worker uvicorn por contenedor: Cloud Run escala horizontalmente,
#     duplicar workers solo duplica la RAM (TF y los 6 modelos se cargarían
#     dos veces). El paralelismo lo da `THREADPOOL_SIZE` + asyncio.
#
# Notas de tamaño:
#   - `opencv-python` requiere libs gráficas (libgl1, libglib2.0-0). Si
#     querés bajar ~150 MB de la imagen, cambiá en `requirements.txt`:
#         opencv-python==4.10.0.84  ->  opencv-python-headless==4.10.0.84
#     y eliminá `libgl1 libglib2.0-0 libsm6 libxext6 libxrender1` de abajo.
#   - `ffmpeg` se queda: OpenCV lo usa para streamear las URLs HTTP de
#     video que vienen en los payloads.
# =============================================================================

# El proyecto se desarrolló sobre Python 3.10.11 por estabilidad con
# MediaPipe. Fijamos el patch version para evitar sorpresas entre builds.
FROM python:3.10.11-slim AS runtime

# ---------- Dependencias del sistema ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------- Variables de entorno globales ----------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    # Cloud Run estándar:
    PORT=8080

WORKDIR /app

# ---------- Dependencias Python ----------
# Copiamos requirements primero para aprovechar el cache de Docker:
# si solo cambia el código, no se reinstala TF/MediaPipe.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ---------- Código + modelos ----------
# `COPY app/ ./app/` copia TODO lo que esté dentro de `app/` y no esté
# excluido por `.dockerignore`: código Python, JSON de catálogo,
# `app/info_models/*.json`, logs y pesos `app/models/*.keras`.
# No se necesita volumen en Cloud Run mientras esos archivos viajen en
# el repositorio/imagen.
COPY app/ ./app/

# ---------- Usuario no-root (recomendado por seguridad en Cloud Run) ----------
RUN useradd --create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app
USER appuser

# ---------- Arranque ----------
EXPOSE 8080
# `exec` evita un proceso shell extra; uvicorn queda como PID 1 y
# recibe SIGTERM directo cuando Cloud Run apaga la instancia.
CMD exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    --timeout-keep-alive 75
