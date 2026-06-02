# app/api/v1/endpoints/testing_models.py
"""Endpoints de prueba para cada modelo ColSign por separado.

Pensados para validar cada modelo de forma aislada sin pasar por el
flujo jerárquico. Todos siguen el mismo contrato:

- Input  : un único `video_url` (video corto de 2-3 s) O una `sequence`
           (45, 225) ya pre-procesada.
- Output : 1 sola predicción.

Endpoints expuestos (con prefix `/test` definido en `router.py`):

    POST /test/raiz         -> grupo morfológico                (4 clases)
    POST /test/static       -> sub-modelo Grupo Estático        (21 clases)
    POST /test/unimanual    -> sub-modelo Grupo Unimanual       (64 clases)
    POST /test/simetrico    -> sub-modelo Bimanual Simétrico    (30 clases)
    POST /test/asimetrico   -> sub-modelo Bimanual Asimétrico   (39 clases)
    POST /test/plano        -> modelo plano (todas las clases)  (154 clases)
"""

from typing import Callable

import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas.colsign import (
    SequenceInput,
    SinglePredictionRequest,
    SinglePredictionResponse,
)
from app.services import (
    pipeline_colsign_model_asimetrico as asimetrico_pipeline,
    pipeline_colsign_model_plano as plano_pipeline,
    pipeline_colsign_model_raiz_v2 as raiz_pipeline,
    pipeline_colsign_model_simetrico as simetrico_pipeline,
    pipeline_colsign_model_static_v2 as static_pipeline,
    pipeline_colsign_model_unimanual_v2 as unimanual_pipeline,
)


router = APIRouter()


# =====================================================================
# Helpers
# =====================================================================

def _sequence_input_to_ndarray(sequence: SequenceInput) -> np.ndarray:
    return np.asarray(sequence.data, dtype=np.float32)


def _run_single(predict_one: Callable, payload: SinglePredictionRequest) -> dict:
    """Despachador común: arma kwargs según el payload y ejecuta `predict_one`
    del sub-pipeline, traduciendo excepciones a códigos HTTP apropiados.
    """
    if payload.sequence is not None:
        kwargs = {"sequence": _sequence_input_to_ndarray(payload.sequence)}
    elif payload.video_url is not None:
        kwargs = {"video_url": payload.video_url}
    else:
        # Imposible por el @model_validator del schema.
        raise HTTPException(status_code=400, detail="Payload inválido.")

    try:
        return predict_one(**kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IOError as exc:
        # p.ej. video no descargable / URL muerta
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado: {type(exc).__name__}: {exc}",
        ) from exc


# =====================================================================
# Modelo raíz
# =====================================================================

@router.post("/raiz", response_model=SinglePredictionResponse)
def test_raiz(payload: SinglePredictionRequest):
    """Modelo raíz: clasifica el GRUPO morfológico de UNA seña (4 clases)."""
    result = _run_single(raiz_pipeline.predict_one, payload)
    return _to_response(result)


# =====================================================================
# Sub-modelos
# =====================================================================

@router.post("/static", response_model=SinglePredictionResponse)
def test_static(payload: SinglePredictionRequest):
    """Sub-modelo Grupo Estático (vocales/consonantes estáticas, 21 clases)."""
    result = _run_single(static_pipeline.predict_one, payload)
    return _to_response(result)


@router.post("/unimanual", response_model=SinglePredictionResponse)
def test_unimanual(payload: SinglePredictionRequest):
    """Sub-modelo Grupo Dinámico Unimanual (64 clases)."""
    result = _run_single(unimanual_pipeline.predict_one, payload)
    return _to_response(result)


@router.post("/simetrico", response_model=SinglePredictionResponse)
def test_simetrico(payload: SinglePredictionRequest):
    """Sub-modelo Grupo Dinámico Bimanual Simétrico (30 clases)."""
    result = _run_single(simetrico_pipeline.predict_one, payload)
    return _to_response(result)


@router.post("/asimetrico", response_model=SinglePredictionResponse)
def test_asimetrico(payload: SinglePredictionRequest):
    """Sub-modelo Grupo Dinámico Bimanual Asimétrico (39 clases)."""
    result = _run_single(asimetrico_pipeline.predict_one, payload)
    return _to_response(result)


# =====================================================================
# Modelo plano (todas las clases, predicción individual)
# =====================================================================

@router.post("/plano", response_model=SinglePredictionResponse)
def test_plano(payload: SinglePredictionRequest):
    """Modelo plano `colsign_lstm_norm_45_154` (154 clases) en modo INDIVIDUAL.

    SIEMPRE devuelve 1 sola predicción sin importar la duración del
    video; muestrea 45 frames uniformemente del video completo. Para
    procesar videos largos como múltiples predicciones usa el endpoint
    batch `/sign-to-text/flat` en `translation.py`.
    """
    result = _run_single(plano_pipeline.predict_one, payload)
    return _to_response(result)


# =====================================================================
# Serialización
# =====================================================================

def _to_response(result: dict) -> SinglePredictionResponse:
    """Filtra campos no serializables (ej. el ndarray `proba`) antes de
    pasar al schema. Hoy `predict_one` solo devuelve `proba` cuando se
    pidió `include_proba=True`, cosa que NO hacen los endpoints; pero
    dejamos el filtro como defensa para futuras extensiones.
    """
    safe = {k: v for k, v in result.items() if k != "proba"}
    return SinglePredictionResponse(**safe)
