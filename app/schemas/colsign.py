"""Schemas Pydantic para los endpoints ColSign (sign-to-text).

Cubre dos familias de endpoints:

- **Batch** (modelo plano y jerárquico, en `translation.py`):
  aceptan una URL, una lista de URLs o una lista de secuencias y
  siempre devuelven `list[Prediction]`.

- **Individuales** (raíz y sub-modelos, en `testing_models.py`):
  aceptan un único video o una única secuencia y devuelven 1 dict.

Las secuencias se serializan como lista de listas de floats con shape
``(45, 225)``, validada en `SequenceInput`. Si en un futuro pesa
demasiado para HTTP, se podría aceptar también un `bytes`/base64 con
un ndarray float32 plano, pero por ahora JSON puro es ~80 KB por
secuencia, perfectamente viable.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.video_processor import (
    DEFAULT_CLIP_SECONDS,
    DEFAULT_MAX_VIDEO_SECONDS,
)


# Las constantes deben coincidir con `inference_helpers.EXPECTED_*`.
# Las repetimos aquí para no acoplar el schema con TensorFlow.
SEQUENCE_FRAMES   = 45
SEQUENCE_FEATURES = 225


# =====================================================================
# Inputs
# =====================================================================

class SequenceInput(BaseModel):
    """Una secuencia de keypoints ya normalizados, lista para predecir.

    Forma esperada: ``(45, 225)`` = 45 frames × 225 features pose+manos
    normalizados con `drop_pose_visibility=True`.
    """

    data: List[List[float]] = Field(
        ...,
        description=(
            f"Matriz de shape ({SEQUENCE_FRAMES}, {SEQUENCE_FEATURES}) con "
            f"keypoints ya normalizados (output de "
            f"`utils_pipeplanes.extract_lstm_features`)."
        ),
    )

    @field_validator("data")
    @classmethod
    def _check_shape(cls, value: List[List[float]]) -> List[List[float]]:
        if len(value) != SEQUENCE_FRAMES:
            raise ValueError(
                f"Se esperaban {SEQUENCE_FRAMES} frames, llegaron {len(value)}."
            )
        for i, row in enumerate(value):
            if len(row) != SEQUENCE_FEATURES:
                raise ValueError(
                    f"Frame {i}: se esperaban {SEQUENCE_FEATURES} features, "
                    f"llegaron {len(row)}."
                )
        return value


class SinglePredictionRequest(BaseModel):
    """Request para los endpoints individuales: un solo video o una sola secuencia."""

    video_url: Optional[str] = Field(
        default=None,
        description="URL pública del video corto (2-3 s).",
    )
    sequence: Optional[SequenceInput] = Field(
        default=None,
        description="Secuencia (45, 225) ya pre-procesada en el cliente.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "SinglePredictionRequest":
        provided = sum(x is not None for x in (self.video_url, self.sequence))
        if provided != 1:
            raise ValueError(
                "Debes enviar EXACTAMENTE uno de: `video_url` o `sequence`."
            )
        return self


class BatchPredictionRequest(BaseModel):
    """Request para los endpoints batch (plano y jerárquico).

    Cualquiera de los tres campos (mutuamente exclusivos):
        - `video_url`  : 1 video → 1 o N predicciones (según duración).
        - `video_urls` : varios videos → resultados concatenados.
        - `sequences`  : lista de secuencias pre-procesadas.
    """

    video_url: Optional[str] = Field(
        default=None,
        description="URL pública de UN video.",
    )
    video_urls: Optional[List[str]] = Field(
        default=None,
        description="Lista de URLs públicas de videos.",
    )
    sequences: Optional[List[SequenceInput]] = Field(
        default=None,
        description="Lista de secuencias (45, 225) ya pre-procesadas.",
    )
    clip_seconds: float = Field(
        default=DEFAULT_CLIP_SECONDS,
        ge=0.5,
        le=DEFAULT_MAX_VIDEO_SECONDS,
        description=(
            "Duración en segundos de cada clip al dividir videos largos "
            f"(>3 s). Default {DEFAULT_CLIP_SECONDS}. Máximo "
            f"{DEFAULT_MAX_VIDEO_SECONDS:.0f} (tope del video). Solo aplica "
            "cuando la entrada es `video_url` o `video_urls`; se ignora "
            "con `sequences`."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "BatchPredictionRequest":
        provided = sum(
            x is not None for x in (self.video_url, self.video_urls, self.sequences)
        )
        if provided != 1:
            raise ValueError(
                "Debes enviar EXACTAMENTE uno de: "
                "`video_url`, `video_urls` o `sequences`."
            )
        return self


# =====================================================================
# Outputs
# =====================================================================

class PredictionResult(BaseModel):
    """Predicción individual (de un sub-modelo o del modelo plano).

    Todos los campos son opcionales para soportar también el caso de
    error (un dict con solo `error`, `source`, `source_index`, ...).
    """

    # Pydantic v2 protege por defecto namespaces que empiezan con `model_*`.
    # Lo desactivamos porque `model_name` es un campo legítimo en nuestra API.
    model_config = ConfigDict(protected_namespaces=())

    label: Optional[str] = Field(default=None, description="Etiqueta predicha.")
    prob: Optional[float] = Field(default=None, description="Probabilidad top-1.")
    id: Optional[int] = Field(default=None, description="ID interno de la clase.")
    model_name: Optional[str] = Field(
        default=None, description="Nombre canónico del modelo usado."
    )
    num_classes: Optional[int] = Field(
        default=None, description="Número total de clases del modelo."
    )

    # Trazabilidad de origen para batches
    source: Optional[str] = Field(
        default=None, description='Fuente del input: "video" o "sequence".'
    )
    source_index: Optional[int] = Field(
        default=None, description="Índice del input en la lista original."
    )
    clip_index: Optional[int] = Field(
        default=None,
        description="Índice del clip dentro del video (0 si era corto o si era secuencia).",
    )
    video: Optional[str] = Field(
        default=None, description="URL/path del video procesado (si aplica)."
    )

    error: Optional[str] = Field(
        default=None,
        description="Mensaje de error si esta predicción específica falló.",
    )


class HierarchicalPredictionResult(PredictionResult):
    """Predicción del pipeline jerárquico, incluye info del paso raíz."""

    grupo_raiz: Optional[str] = Field(
        default=None, description="Grupo morfológico predicho por el modelo raíz."
    )
    grupo_raiz_prob: Optional[float] = Field(
        default=None, description="Probabilidad asignada por el raíz al grupo elegido."
    )
    grupo_raiz_model: Optional[str] = Field(
        default=None, description="Nombre canónico del modelo raíz."
    )


class FlatBatchResponse(BaseModel):
    """Respuesta del endpoint plano (`/sign-to-text/flat`)."""

    count: int = Field(..., description="Cantidad de predicciones devueltas.")
    predictions: List[PredictionResult]


class HierarchicalBatchResponse(BaseModel):
    """Respuesta del endpoint jerárquico (`/sign-to-text/hierarchical`)."""

    count: int = Field(..., description="Cantidad de predicciones devueltas.")
    predictions: List[HierarchicalPredictionResult]


class SinglePredictionResponse(PredictionResult):
    """Respuesta de los endpoints individuales en `testing_models.py`."""
    pass


# =====================================================================
# Narrativa (Gemini sobre la lista de señas predichas)
# =====================================================================

class NarrativeSenseRequest(BaseModel):
    """Request del endpoint `/narrative-sense`.

    Recibe una lista de señas crudas (etiquetas) detectadas en orden
    secuencial y Gemini las convierte en una oración natural en español.
    """

    words: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "Lista ordenada de señas predichas. "
            "Ej: ['Hola', 'd', 'i', 'e', 'g', 'o']."
        ),
    )


# Default sugerido para el filtro de confianza antes de pasar etiquetas a
# Gemini. Estricto: solo predicciones muy seguras llegan al LLM. El cliente
# puede sobrescribirlo por request (ej. 0.0 para deshabilitar el filtro).
DEFAULT_MIN_CONFIDENCE = 0.7


class NarrativeBatchPredictionRequest(BatchPredictionRequest):
    """Request para los endpoints `/sign-to-narrative/*`.

    Extiende `BatchPredictionRequest` agregando un filtro de confianza
    que se aplica ANTES de enviar las etiquetas a Gemini: las predicciones
    con `prob` menor a `min_confidence` se descartan. Esto:

    - Evita contaminar la narrativa con clips de baja confianza (típicamente
      frames de transición donde el modelo "adivina").
    - Ahorra cuota de Gemini cuando muchos clips son ruido.
    - No afecta a `predictions` en la respuesta (siguen llegando todas);
      solo filtra el INPUT al LLM.

    El default (`0.7`) es estricto; baja a `0.5` o `0.3` si necesitas
    capturar señas con mala iluminación/encuadre. Pasa `0.0` para
    deshabilitar el filtro por completo.
    """

    min_confidence: float = Field(
        default=DEFAULT_MIN_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description=(
            "Probabilidad mínima top-1 para que una predicción se envíe "
            "a Gemini. Default 0.7 (estricto). 0.0 deshabilita el filtro."
        ),
    )


class NarrativeSenseResponse(BaseModel):
    """Respuesta del endpoint `/narrative-sense`."""

    words: List[str] = Field(..., description="Lista original de señas recibida.")
    narrative: str = Field(..., description="Oración o párrafo en español natural.")


class FlatNarrativeResponse(FlatBatchResponse):
    """Respuesta del endpoint combinado `/sign-to-narrative/flat`.

    Igual que `FlatBatchResponse` pero añade el resultado de pasar las
    etiquetas predichas por Gemini para obtener una narrativa fluida.
    """

    narrative: Optional[str] = Field(
        default=None,
        description=(
            "Oración natural generada por Gemini a partir de las "
            "etiquetas predichas. `null` si no hubo predicciones válidas."
        ),
    )


class HierarchicalNarrativeResponse(HierarchicalBatchResponse):
    """Respuesta del endpoint combinado `/sign-to-narrative/hierarchical`."""

    narrative: Optional[str] = Field(
        default=None,
        description=(
            "Oración natural generada por Gemini a partir de las "
            "etiquetas predichas. `null` si no hubo predicciones válidas."
        ),
    )
