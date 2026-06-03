# Colsign API Models 🤟🤖

Backend de modelos ILSC del sistema **Colsign** https://www.colsign.com.co/, una plataforma de investigación aplicada para la inclusión social de la comunidad sorda. Orquesta la traducción bidireccional entre **texto** y **Lengua de Señas Colombiana (LSC)** combinando modelos LSTM propios (Keras/TensorFlow) con Google Gemini.

Construido con **FastAPI**, expone:

- **Sign → Text**: 6 modelos LSTM (1 raíz + 4 sub-modelos + 1 plano) que reciben videos públicos o secuencias `(45, 225)` ya pre-procesadas y devuelven predicciones de señas.
- **Sign → Narrativa**: pipeline combinado que pasa las predicciones a Gemini para producir una oración fluida en español.
- **Text → Sign**: traducción de texto libre a la lista de señas del catálogo oficial (566 señas), usando Structured Outputs de Gemini.

---

## Características principales

- **Doble estrategia de inferencia**:
  - **Pipeline jerárquico v2**: un modelo *raíz* clasifica el grupo morfológico (Estático, Unimanual, Bimanual Simétrico, Bimanual Asimétrico) y delega la etiqueta final al sub-modelo correspondiente.
  - **Pipeline plano**: un único modelo de 154 clases como fallback / baseline.
- **Procesamiento de video integrado**: OpenCV (vía FFmpeg) descarga el stream de la URL, calcula FPS efectivo, parte en clips contiguos de 2 s y extrae keypoints normalizados con MediaPipe Holistic (`pose + manos`, 225 features).
- **Tope de duración (60 s) en dos pasos**: pre-check por metadata + tope defensivo durante la lectura → HTTP 413 con mensaje estructurado en los endpoints individuales.
- **Sentido narrativo con Gemini** (`gemini-2.5-flash`), con filtro de confianza configurable (`min_confidence`, default 0.7) para no enviar predicciones ruidosas al LLM y ahorrar cuota.
- **Concurrencia para 50+ usuarios**: endpoints `async`, llamadas a Gemini con el cliente async del SDK, inferencia TF/OpenCV en `run_in_threadpool`, y threadpool de AnyIO subido a 64 tokens en el `lifespan`.
- **Catálogo y modelos pre-cargados** en el `lifespan` para que la primera request no pague el coste de I/O ni de inicialización de TF.
- **Trazabilidad por predicción**: cada resultado lleva `source`, `source_index`, `clip_index`, `video`, `model_name` y `num_classes` para auditar batches mixtos.

---

## Stack tecnológico

| Capa | Tecnología |
|------|------------|
| Core | Python 3.11+, FastAPI 0.111 |
| Validación | Pydantic v2 + Pydantic Settings |
| Inferencia | TensorFlow 2.17 + Keras 3.5 |
| Visión | OpenCV 4.10 + MediaPipe Holistic 0.10 |
| LLM | Google GenAI SDK 2.6 (`gemini-2.5-flash`) |
| Servidor | Uvicorn 0.30 |

Versiones exactas en [`requirements.txt`](./requirements.txt).

---

## Estructura del proyecto

```text
api_models_lsc/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── translation.py        # Texto↔señas, batch y narrativa (Gemini)
│   │       │   └── testing_models.py     # Endpoints individuales por modelo (/test/*)
│   │       └── router.py                 # Composición de routers v1
│   ├── core/
│   │   ├── config.py                     # Settings (env vars, rutas, threadpool)
│   │   ├── lifespan.py                   # Carga de modelos y catálogo en arranque
│   │   └── model_registry.py             # API única para resolver/cargar modelos
│   ├── info_models/                      # Metadata por modelo
│   │   ├── colsign_lstm_norm_45_154_labels.json
│   │   ├── colsign_lstm_norm_raiz_45_154_v2_labels.json
│   │   ├── colsign_lstm_norm_estatic_45_154_v2_labels.json
│   │   ├── colsign_lstm_norm_unimanual_45_154_v2_labels.json
│   │   ├── colsign_lstm_norm_bi_simetrico_45_154_labels.json
│   │   ├── colsign_lstm_norm_bi_asimetrico_45_154_labels.json
│   │   └── *_train_log.txt               # Logs de entrenamiento (referencia)
│   ├── information_json/
│   │   └── name_sign_list.json           # Catálogo oficial de 566 señas (Text→Sign)
│   ├── models/                           # Checkpoints Keras (LSTM v2 y plano)
│   │   ├── colsign_lstm_norm_45_154_best.keras
│   │   ├── colsign_lstm_norm_raiz_45_154_v2_best.keras
│   │   ├── colsign_lstm_norm_estatic_45_154_v2_best.keras
│   │   ├── colsign_lstm_norm_unimanual_45_154_v2_best.keras
│   │   ├── colsign_lstm_norm_bi_simetrico_45_154_best.keras
│   │   └── colsign_lstm_norm_bi_asimetrico_45_154_best.keras
│   ├── models_lsc/                       # Modelos legacy .h5 (compatibilidad)
│   ├── schemas/
│   │   ├── colsign.py                    # Schemas para sign-to-text + narrativa
│   │   └── translation.py                # Schemas para text-to-sign
│   ├── services/
│   │   ├── gemini_service.py             # Cliente async de Gemini
│   │   ├── video_processor.py            # OpenCV + clips + MediaPipe
│   │   ├── pipeline_colsign_45_154.py    # Pipeline plano (batch)
│   │   ├── pipeline_colsign_jerarquico_v2.py # Pipeline jerárquico (batch)
│   │   ├── pipeline_colsign_model_raiz_v2.py    # Modelo raíz (individual)
│   │   ├── pipeline_colsign_model_static_v2.py  # Sub-modelo estático
│   │   ├── pipeline_colsign_model_unimanual_v2.py
│   │   ├── pipeline_colsign_model_simetrico.py
│   │   ├── pipeline_colsign_model_asimetrico.py
│   │   ├── pipeline_colsign_model_plano.py      # Plano (predicción individual)
│   │   └── src/
│   │       ├── utils_pipeplanes.py       # Carga Keras + extract_lstm_features
│   │       └── inference_helpers.py      # Resolución de modelos + predict_sequence
│   └── main.py                           # Entry point FastAPI
├── .env                                  # Variables sensibles (ignorado en git)
└── requirements.txt
```

---

## Catálogo de modelos LSTM v2

Todos los modelos comparten input shape **`(45, 225)`** — 45 frames muestreados uniformemente × 225 features (pose 33×4 sin visibility + 2 manos × 21 × 3, todo normalizado). Todo lo relacionado al entrenamiento lo encuentran en: https://github.com/FerchoRV/train-models-colsign-154-3659 

| Nombre canónico | Rol | Clases |
|---|---|---|
| `colsign_lstm_norm_raiz_45_154_v2` | Raíz: predice el **grupo morfológico** | 4 |
| `colsign_lstm_norm_estatic_45_154_v2` | Sub-modelo **Grupo Estático** | 21 |
| `colsign_lstm_norm_unimanual_45_154_v2` | Sub-modelo **Grupo Dinámico Unimanual** | 64 |
| `colsign_lstm_norm_bi_simetrico_45_154` | Sub-modelo **Bimanual Simétrico** | 30 |
| `colsign_lstm_norm_bi_asimetrico_45_154` | Sub-modelo **Bimanual Asimétrico** | 39 |
| `colsign_lstm_norm_45_154` | **Plano** (todas las clases en una sola red) | 154 |

Cada modelo se descubre a partir del nombre canónico: `app/models/<name>_best.keras` + `app/info_models/<name>_labels.json`. La resolución vive en `app/services/src/utils_pipeplanes.py` y se accede a través de `app/core/model_registry.py`.

---

## Configuración e instalación

### 1. Entorno virtual

```bash
python3.11 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
```

### 2. Dependencias

```bash
pip install -r requirements.txt
```

### 3. Variables de entorno (`.env`)

```env
GEMINI_API_KEY="TU_API_KEY_DE_GOOGLE_AI_STUDIO"
```

### 4. Modelos

Los seis `*_best.keras` deben estar en `app/models/` y su metadata correspondiente en `app/info_models/`. Si falta el plano, el sistema sigue arrancando y solo deshabilita los endpoints planos; si falta el raíz, el `lifespan` lo reporta como advertencia en `/`.

### 5. Ejecutar el servidor

```bash
uvicorn app.main:app --reload
```

Documentación interactiva (Swagger UI): <http://127.0.0.1:8000/docs>
Health check: `GET /` devuelve `{ "api": ..., "model_status": ... }`.

---

## Endpoints (`/api/v1`)

### Texto → Señas (Gemini)

`POST /text-to-sign` — Traduce texto libre a una lista ordenada de señas del catálogo oficial (566) usando Structured Outputs.

```json
{ "text": "Hola, ¿cómo estás? Yo quiero comer" }
```

```json
{
  "text_original": "Hola, ¿cómo estás? Yo quiero comer",
  "señas_seleccionadas": ["Hola", "Cómo estás", "Yo", "Querer", "c", "o", "m", "e", "r"]
}
```

---

### Señas → Texto (batch)

Ambos endpoints aceptan exactamente UNO de: `video_url`, `video_urls` o `sequences`.

- `POST /sign-to-text/flat` → modelo plano de 154 clases.
- `POST /sign-to-text/hierarchical` → pipeline jerárquico v2 (raíz → sub-modelo).

**Reglas de inferencia por video**:
- ≤ 3 s → 1 predicción (todo el video como un único clip de 45 frames).
- > 3 s → N predicciones, cortando en clips contiguos de ~2 s. El residuo final se descarta si tiene menos de 45 frames (no alcanzaría para muestrear sin repetir).
- > 60 s → rechazado (`VideoTooLongError`).

```json
{ "video_url": "https://cdn.colsign.com.co/clips/hola.mp4" }
```

```json
{
  "count": 2,
  "predictions": [
    {
      "label": "Hola", "prob": 0.94, "id": 12,
      "model_name": "colsign_lstm_norm_45_154", "num_classes": 154,
      "source": "video", "source_index": 0, "clip_index": 0,
      "video": "https://cdn.colsign.com.co/clips/hola.mp4"
    },
    {
      "label": "Gracias", "prob": 0.88, "id": 47,
      "model_name": "colsign_lstm_norm_45_154", "num_classes": 154,
      "source": "video", "source_index": 0, "clip_index": 1,
      "video": "https://cdn.colsign.com.co/clips/hola.mp4"
    }
  ]
}
```

El endpoint jerárquico añade además `grupo_raiz`, `grupo_raiz_prob` y `grupo_raiz_model` por predicción para trazar la decisión del raíz.

---

### Narrativa con Gemini

- `POST /narrative-sense` — Recibe `{ "words": ["Hola", "d", "i", "e", "g", "o"] }` y devuelve `{ "narrative": "Hola Diego." }`. Útil cuando el cliente ya tiene las predicciones acumuladas y solo quiere la oración.
- `POST /sign-to-narrative/flat` — `/sign-to-text/flat` + Gemini en una sola request.
- `POST /sign-to-narrative/hierarchical` — `/sign-to-text/hierarchical` + Gemini en una sola request.

Los combinados aceptan además `min_confidence` (default `0.7`). Las predicciones con `prob < min_confidence` **no** se envían a Gemini; siguen apareciendo íntegras en `predictions` pero no contaminan la narrativa. Si tras filtrar no queda ninguna etiqueta válida, `narrative` es `null` y **no** se consume cuota de Gemini.

```json
{
  "video_url": "https://cdn.colsign.com.co/clips/hola-diego.mp4",
  "clip_seconds": 1.5,
  "min_confidence": 0.6
}
```

```json
{
  "count": 6,
  "predictions": [/* ... seis predicciones con sus prob ... */],
  "narrative": "Hola Diego."
}
```

---

### Endpoints individuales (`/test/*`)

Hechos para validar cada modelo de forma aislada (no para producción batch). Reciben **un único** `video_url` o `sequence` y devuelven **1 sola** predicción.

| Endpoint | Modelo subyacente |
|---|---|
| `POST /test/raiz` | Raíz (4 grupos) |
| `POST /test/static` | Sub-modelo Estático |
| `POST /test/unimanual` | Sub-modelo Unimanual |
| `POST /test/simetrico` | Sub-modelo Bimanual Simétrico |
| `POST /test/asimetrico` | Sub-modelo Bimanual Asimétrico |
| `POST /test/plano` | Plano (154 clases) en modo individual: muestrea 45 frames del video completo, **siempre 1 predicción** sin importar duración (mientras no se exceda el tope de 60 s). |

```json
{ "video_url": "https://cdn.colsign.com.co/clips/letra-a.mp4" }
```

o bien con secuencia ya pre-procesada:

```json
{ "sequence": { "data": [[...225 floats...], /* x45 */] } }
```

---

## Códigos de error relevantes

| Código | Causa |
|---|---|
| `400` | Payload inválido (shape errónea, ambos campos enviados, etc.) |
| `413` | Video excede `DEFAULT_MAX_VIDEO_SECONDS` (60 s). El detail incluye `actual_seconds` y `max_seconds`. |
| `422` | OpenCV no logró abrir la fuente (URL muerta, formato no soportado). |
| `502` | Gemini devolvió error en `/narrative-sense`. |
| `503` | Modelo solicitado no disponible en disco. |
| `500` | Excepción inesperada (con tipo y mensaje en el detail). |

En los endpoints batch, errores por video individual no rompen toda la response: cada slot fallido se devuelve como `{ "error": "TipoError: mensaje", ... }` dentro de `predictions`.

---

## Notas de concurrencia y despliegue

- Los endpoints que hablan con Gemini son `async` y usan `client.aio.models.generate_content`, así no bloquean el event loop durante los 200–1500 ms típicos del LLM.
- Los pipelines TF/OpenCV/MediaPipe son síncronos; siempre se invocan vía `fastapi.concurrency.run_in_threadpool` desde endpoints `async`.
- El `lifespan` ajusta el threadpool de AnyIO a `settings.THREADPOOL_SIZE = 64` (default del SDK suele ser ~6–8 en Cloud Run con 2 CPU). Esto sube el techo de concurrencia sin afectar coste; el límite real pasa a ser CPU del contenedor.
- MediaPipe Holistic se instancia **una sola vez por video** y se reutiliza entre clips para evitar el coste de inicialización repetida.
- `utils_pipeplanes.load_model` cachea por ruta `.keras`, así que los `model_registry.load_*` no recargan pesos en llamadas sucesivas.

---

## 🤝 Contribuciones

Proyecto de investigación aplicada para inclusión social. Son bienvenidas mejoras en:

- Optimización del pipeline de extracción de keypoints (latencia OpenCV/MediaPipe).
- Estrategias de muestreo temporal alternativas a `np.linspace(45)`.
- Heurísticas de filtrado pre-Gemini (más allá del umbral de confianza fijo).
- Métricas de observabilidad (latencia por etapa: download → frames → keypoints → predict → Gemini).
