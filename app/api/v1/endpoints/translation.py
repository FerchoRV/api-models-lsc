# app/api/v1/endpoints/translation.py
"""Endpoints de traducción.

- `POST /text-to-sign`                    : texto → lista de señas (vía Gemini).
- `POST /sign-to-text/flat`               : video(s) o secuencias → predicciones
                                            con el modelo plano de 154 clases.
- `POST /sign-to-text/hierarchical`       : video(s) o secuencias → predicciones
                                            con el pipeline jerárquico (raíz +
                                            sub-modelo).
- `POST /narrative-sense`                 : lista de señas crudas → oración
                                            con sentido narrativo (Gemini).
- `POST /sign-to-narrative/flat`          : video(s) → predicciones + narrativa
                                            en una sola llamada (modelo plano).
- `POST /sign-to-narrative/hierarchical`  : video(s) → predicciones + narrativa
                                            en una sola llamada (jerárquico).

Notas de concurrencia
---------------------
Los endpoints que tocan Gemini son ``async def`` y usan el cliente async
del SDK (``client.aio.models.generate_content``) para no bloquear el
event loop mientras el LLM responde (200-1500 ms típico).

Los pipelines de inferencia (TensorFlow + OpenCV + MediaPipe) son
SÍNCRONOS y bloqueantes, así que cuando se invocan desde un endpoint
``async def`` los envolvemos con ``run_in_threadpool`` para que no
bloqueen el event loop tampoco. Esto preserva el beneficio del async:
con 50 usuarios concurrentes, FastAPI puede mantener muchas requests
"en vuelo" sin saturar el threadpool con awaits a Gemini.
"""

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.schemas.colsign import (
    BatchPredictionRequest,
    FlatBatchResponse,
    FlatNarrativeResponse,
    HierarchicalBatchResponse,
    HierarchicalNarrativeResponse,
    NarrativeBatchPredictionRequest,
    NarrativeSenseRequest,
    NarrativeSenseResponse,
    SequenceInput,
    SlidingWindowNarrativeRequest,
    WindowNarrativeResponse,
)
from app.schemas.translation import TextToSignRequest, TextToSignResponse
from app.services import pipeline_colsign_45_154 as flat_pipeline
from app.services import pipeline_colsign_jerarquico_v2 as hierarchical_pipeline
from app.services import video_windows_processor as windows
from app.services.gemini_service import GeminiService
from app.services.video_processor import VideoTooLongError


router = APIRouter()
gemini_service = GeminiService()


# =====================================================================
# Texto → Señas (Gemini)
# =====================================================================

@router.post("/text-to-sign", response_model=TextToSignResponse)
async def text_to_sign(payload: TextToSignRequest, request: Request):
    catalogo_nombres = getattr(request.app.state, "name_sign_list", [])

    if not catalogo_nombres:
        raise HTTPException(
            status_code=500,
            detail="El catálogo de señas no está disponible o no se pudo cargar en el servidor.",
        )

    señas_elegidas = await gemini_service.select_signs_from_text(
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
async def sign_to_text_flat(payload: BatchPredictionRequest):
    """Predice una o varias señas usando el modelo plano `colsign_lstm_norm_45_154`.

    - `video_url` único: 1 predicción si dura ≤ 3 s, N si dura más (clips de
      `clip_seconds`, default 1.5 s, máx. 60).
    - `video_urls`: cada URL aporta sus propias predicciones, concatenadas.
    - `sequences`: 1 predicción por cada secuencia (45, 225) enviada.
    """
    predictions = await _run_pipeline_in_threadpool(flat_pipeline.predict, payload)
    return FlatBatchResponse(count=len(predictions), predictions=predictions)


# =====================================================================
# Señas → Texto (pipeline jerárquico: raíz + sub-modelo)
# =====================================================================

@router.post("/sign-to-text/hierarchical", response_model=HierarchicalBatchResponse)
async def sign_to_text_hierarchical(payload: BatchPredictionRequest):
    """Predice una o varias señas usando el pipeline jerárquico v2.

    Flujo por cada secuencia: modelo raíz determina el grupo morfológico
    y luego el sub-modelo correspondiente produce la etiqueta final.

    Mismas opciones de entrada que `/sign-to-text/flat`.
    """
    predictions = await _run_pipeline_in_threadpool(hierarchical_pipeline.predict, payload)
    return HierarchicalBatchResponse(count=len(predictions), predictions=predictions)


# =====================================================================
# Narrativa: lista cruda de señas → oración con sentido (Gemini)
# =====================================================================

@router.post("/narrative-sense", response_model=NarrativeSenseResponse)
async def narrative_sense(payload: NarrativeSenseRequest):
    """Convierte una lista de señas predichas en una oración natural.

    Útil cuando el cliente ya tiene las predicciones (por ejemplo, las
    obtuvo de `/sign-to-text/*` o las acumuló desde varios endpoints
    individuales) y solo necesita el ajuste narrativo.
    """
    try:
        narrative = await gemini_service.get_narrative_sense(payload.words)
    except Exception as exc:  # noqa: BLE001 - barrera contra fallos del SDK de Gemini
        raise HTTPException(
            status_code=502,
            detail=f"Error consultando a Gemini: {type(exc).__name__}: {exc}",
        ) from exc

    return NarrativeSenseResponse(words=payload.words, narrative=narrative)


# =====================================================================
# Señas → Narrativa (combinado: video → predicciones + Gemini)
# =====================================================================

@router.post("/sign-to-narrative/flat", response_model=FlatNarrativeResponse)
async def sign_to_narrative_flat(payload: NarrativeBatchPredictionRequest):
    """Combina `/sign-to-text/flat` + `/narrative-sense` en una sola request.

    Ahorra un round-trip de red al cliente y devuelve simultáneamente la
    lista de predicciones (modelo plano) y la oración con sentido
    narrativo generada por Gemini.

    El filtro `min_confidence` (default 0.7) descarta predicciones de
    baja confianza ANTES de pasarlas a Gemini, sin afectar a la lista
    `predictions` que se retorna (siguen llegando todas, con su `prob`).

    Si después de filtrar no quedan etiquetas válidas, `narrative` será
    `null` y NO se gasta una llamada a Gemini.
    """
    predictions = await _run_pipeline_in_threadpool(flat_pipeline.predict, payload)
    narrative = await _build_narrative(predictions, payload.min_confidence)
    return FlatNarrativeResponse(
        count=len(predictions),
        predictions=predictions,
        narrative=narrative,
    )


@router.post("/sign-to-narrative/hierarchical", response_model=HierarchicalNarrativeResponse)
async def sign_to_narrative_hierarchical(payload: NarrativeBatchPredictionRequest):
    """Combina `/sign-to-text/hierarchical` + `/narrative-sense` en una sola request.

    Mismo concepto que `/sign-to-narrative/flat` pero usando el pipeline
    jerárquico (modelo raíz + sub-modelo) que tiende a ser más preciso.
    Acepta el mismo `min_confidence` para filtrar predicciones antes de
    enviarlas a Gemini.
    """
    predictions = await _run_pipeline_in_threadpool(hierarchical_pipeline.predict, payload)
    narrative = await _build_narrative(predictions, payload.min_confidence)
    return HierarchicalNarrativeResponse(
        count=len(predictions),
        predictions=predictions,
        narrative=narrative,
    )


# =====================================================================
# Señas → Narrativa con VENTANA DESLIZANTE (video continuo)
# =====================================================================

@router.post("/sign-to-narrative/flat/sliding-window", response_model=WindowNarrativeResponse)
async def sign_to_narrative_flat_window(payload: SlidingWindowNarrativeRequest):
    """Igual que `/sign-to-narrative/flat` pero con ventana de tiempo deslizante.

    Pensado para VIDEO CONTINUO (varias señas en una grabación). Recorre el
    video con una ventana de `window_seconds` avanzando `stride_seconds`,
    descarta ventanas con confianza < `min_confidence`, penaliza las
    detecciones repetidas (fusiona la misma seña dentro de
    `repeat_gap_seconds` quedándose con la de mayor confianza) e ignora los
    últimos `discard_tail_seconds`. Usa el modelo PLANO de 154 clases.
    """
    predictions = await _run_window_pipeline_in_threadpool(windows.predict_flat, payload)
    narrative = await _build_narrative(predictions, payload.min_confidence)
    return WindowNarrativeResponse(
        count=len(predictions),
        predictions=predictions,
        narrative=narrative,
    )


@router.post(
    "/sign-to-narrative/hierarchical/sliding-window",
    response_model=WindowNarrativeResponse,
)
async def sign_to_narrative_hierarchical_window(payload: SlidingWindowNarrativeRequest):
    """Igual que el anterior pero con el pipeline JERÁRQUICO (raíz + sub-modelo)."""
    predictions = await _run_window_pipeline_in_threadpool(
        windows.predict_hierarchical, payload,
    )
    narrative = await _build_narrative(predictions, payload.min_confidence)
    return WindowNarrativeResponse(
        count=len(predictions),
        predictions=predictions,
        narrative=narrative,
    )


# =====================================================================
# Helpers
# =====================================================================

async def _run_window_pipeline_in_threadpool(
    window_predict_fn,
    payload: SlidingWindowNarrativeRequest,
) -> List[dict]:
    """Ejecuta un `predict_flat/predict_hierarchical` de ventana deslizante
    en el threadpool y traduce excepciones a HTTPException.
    """
    kwargs = {
        "window_seconds":       payload.window_seconds,
        "stride_seconds":       payload.stride_seconds,
        "min_confidence":       payload.min_confidence,
        "repeat_gap_seconds":   payload.repeat_gap_seconds,
        "discard_tail_seconds": payload.discard_tail_seconds,
        "min_segment_frames":   payload.min_segment_frames,
    }
    if payload.video_url is not None:
        kwargs["video_url"] = payload.video_url
    else:
        kwargs["video_urls"] = payload.video_urls

    try:
        return await run_in_threadpool(window_predict_fn, **kwargs)
    except VideoTooLongError as exc:
        raise HTTPException(
            status_code=413,
            detail={
                "message": str(exc),
                "actual_seconds": round(exc.actual_seconds, 2),
                "max_seconds": exc.max_seconds,
            },
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IOError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - barrera de seguridad
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado: {type(exc).__name__}: {exc}",
        ) from exc

async def _run_pipeline_in_threadpool(
    pipeline_predict_fn,
    payload: BatchPredictionRequest,
) -> List[dict]:
    """Ejecuta un `pipeline.predict(...)` síncrono en el threadpool de
    FastAPI y traduce excepciones a HTTPException con códigos útiles.

    Es esencial cuando el endpoint es ``async def``: ejecutar TF/OpenCV
    directamente bloquearía el event loop y mataría la concurrencia.
    """
    try:
        kwargs = _batch_payload_to_kwargs(payload)
        return await run_in_threadpool(pipeline_predict_fn, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - barrera de seguridad
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado: {type(exc).__name__}: {exc}",
        ) from exc


async def _build_narrative(
    predictions: List[dict],
    min_confidence: float = 0.7,
) -> Optional[str]:
    """Extrae etiquetas válidas de las predicciones y pide a Gemini la
    oración con sentido narrativo.

    Reglas de filtrado (en orden):
        1. Descarta entradas con `error` o `label` nula.
        2. Descarta entradas con `prob` menor a `min_confidence`. Si
           `prob` no viene en el dict (p.ej. predicciones de un schema
           viejo), se asume 0.0 y se descarta a menos que `min_confidence`
           sea exactamente 0.0.
        3. Si no quedan etiquetas, devuelve `None` y NO llama a Gemini.

    Si Gemini falla, devuelve `None` (no rompe la respuesta principal con
    las predicciones, que ya fueron calculadas con éxito).
    """
    labels: List[str] = []
    for p in predictions:
        if p.get("error") or p.get("label") is None:
            continue
        prob = p.get("prob")
        if prob is None:
            # Sin probabilidad disponible: solo lo dejamos pasar si el
            # filtro está completamente deshabilitado.
            if min_confidence > 0.0:
                continue
        elif float(prob) < min_confidence:
            continue
        labels.append(p["label"])

    if not labels:
        return None

    try:
        return await gemini_service.get_narrative_sense(labels)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Gemini falló en /sign-to-narrative: {exc}")
        return None


def _batch_payload_to_kwargs(payload: BatchPredictionRequest) -> dict:
    """Convierte el payload validado a los kwargs que esperan los pipelines.

    Las `sequences` (lista de `SequenceInput`) se transforman a una
    lista de ndarrays float32 (45, 225), que es lo que consume
    `predict_sequence` internamente.

    ``clip_seconds`` se propaga a los pipelines cuando el input es video;
    si la entrada son secuencias, el pipeline lo ignora (sin efecto).
    """
    clip_seconds = payload.clip_seconds
    if payload.video_url is not None:
        return {
            "video_url": payload.video_url,
            "clip_seconds": clip_seconds,
        }
    if payload.video_urls is not None:
        return {
            "video_urls": payload.video_urls,
            "clip_seconds": clip_seconds,
        }
    if payload.sequences is not None:
        return {"sequences": _sequences_to_ndarrays(payload.sequences)}
    # Imposible llegar aquí por el @model_validator del schema.
    raise ValueError("Payload inválido: no se reconoció ninguna fuente.")


def _sequences_to_ndarrays(sequences: List[SequenceInput]) -> List[np.ndarray]:
    return [np.asarray(s.data, dtype=np.float32) for s in sequences]
