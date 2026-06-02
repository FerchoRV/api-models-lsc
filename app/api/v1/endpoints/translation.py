# app/api/v1/endpoints/translation.py
"""Endpoints de traducción.

- `POST /text-to-sign`               : texto → lista de señas (vía Gemini).
- `POST /sign-to-text/flat`          : video(s) o secuencias → predicciones
                                       con el modelo plano de 154 clases.
- `POST /sign-to-text/hierarchical`  : video(s) o secuencias → predicciones
                                       con el pipeline jerárquico (raíz +
                                       sub-modelo).
"""

from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.schemas.colsign import (
    BatchPredictionRequest,
    FlatBatchResponse,
    HierarchicalBatchResponse,
    SequenceInput,
)
from app.schemas.translation import TextToSignRequest, TextToSignResponse
from app.services import pipeline_colsign_45_154 as flat_pipeline
from app.services import pipeline_colsign_jerarquico_v2 as hierarchical_pipeline
from app.services.gemini_service import GeminiService


router = APIRouter()
gemini_service = GeminiService()


# =====================================================================
# Texto → Señas (Gemini)
# =====================================================================

@router.post("/text-to-sign", response_model=TextToSignResponse)
def text_to_sign(payload: TextToSignRequest, request: Request):
    catalogo_nombres = getattr(request.app.state, "name_sign_list", [])

    if not catalogo_nombres:
        raise HTTPException(
            status_code=500,
            detail="El catálogo de señas no está disponible o no se pudo cargar en el servidor.",
        )

    señas_elegidas = gemini_service.select_signs_from_text(
        text=payload.text,
        available_signs=catalogo_nombres,
    )

    return TextToSignResponse(
        text_original=payload.text,
        señas_seleccionadas=señas_elegidas,
    )


# =====================================================================
# Señas → Texto (modelo plano de 154 clases)
# =====================================================================

@router.post("/sign-to-text/flat", response_model=FlatBatchResponse)
def sign_to_text_flat(payload: BatchPredictionRequest):
    """Predice una o varias señas usando el modelo plano `colsign_lstm_norm_45_154`.

    - `video_url` único: 1 predicción si dura ≤ 3 s, N si dura más (clips de 2 s).
    - `video_urls`: cada URL aporta sus propias predicciones, concatenadas.
    - `sequences`: 1 predicción por cada secuencia (45, 225) enviada.
    """
    try:
        kwargs = _batch_payload_to_kwargs(payload)
        predictions = flat_pipeline.predict(**kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - barrera de seguridad para no propagar 500 sin contexto
        raise HTTPException(
            status_code=500, detail=f"Error inesperado: {type(exc).__name__}: {exc}"
        ) from exc

    return FlatBatchResponse(count=len(predictions), predictions=predictions)


# =====================================================================
# Señas → Texto (pipeline jerárquico: raíz + sub-modelo)
# =====================================================================

@router.post("/sign-to-text/hierarchical", response_model=HierarchicalBatchResponse)
def sign_to_text_hierarchical(payload: BatchPredictionRequest):
    """Predice una o varias señas usando el pipeline jerárquico v2.

    Flujo por cada secuencia: modelo raíz determina el grupo morfológico
    y luego el sub-modelo correspondiente produce la etiqueta final.

    Mismas opciones de entrada que `/sign-to-text/flat`.
    """
    try:
        kwargs = _batch_payload_to_kwargs(payload)
        predictions = hierarchical_pipeline.predict(**kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Error inesperado: {type(exc).__name__}: {exc}"
        ) from exc

    return HierarchicalBatchResponse(count=len(predictions), predictions=predictions)


# =====================================================================
# Helpers
# =====================================================================

def _batch_payload_to_kwargs(payload: BatchPredictionRequest) -> dict:
    """Convierte el payload validado a los kwargs que esperan los pipelines.

    Las `sequences` (lista de `SequenceInput`) se transforman a una
    lista de ndarrays float32 (45, 225), que es lo que consume
    `predict_sequence` internamente.
    """
    if payload.video_url is not None:
        return {"video_url": payload.video_url}
    if payload.video_urls is not None:
        return {"video_urls": payload.video_urls}
    if payload.sequences is not None:
        return {"sequences": _sequences_to_ndarrays(payload.sequences)}
    # Imposible llegar aquí por el @model_validator del schema.
    raise ValueError("Payload inválido: no se reconoció ninguna fuente.")


def _sequences_to_ndarrays(sequences: List[SequenceInput]) -> List[np.ndarray]:
    return [np.asarray(s.data, dtype=np.float32) for s in sequences]
