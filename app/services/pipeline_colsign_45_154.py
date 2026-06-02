"""Servicio de inferencia del modelo plano `colsign_lstm_norm_45_154`.

Este pipeline corre el modelo de 154 clases (todas las señas en una
sola red) sobre una o varias secuencias. Aceptar varias secuencias en
una sola llamada permite procesar videos largos: se dividen en clips
de ~2 s y cada clip produce una predicción independiente.

API
---
``predict(video_url=None, video_urls=None, sequences=None) -> list[dict]``

Solo UNO de los tres argumentos debe pasarse.

- ``video_url``  : URL pública (o path local) de UN video. Si dura ≤3s
                   genera 1 predicción; si dura más, se divide en
                   clips de 2s y devuelve N predicciones.
- ``video_urls`` : lista de URLs/paths. Cada uno se procesa como arriba
                   y todos los resultados se concatenan en orden.
- ``sequences``  : lista de ndarray ``(45, 225)`` ya normalizados;
                   1 predicción por secuencia.

Cada predicción es un dict con:
    {
        "label":       str,      # seña predicha
        "prob":        float,    # probabilidad top-1
        "id":          int,
        "model_name":  str,
        "num_classes": int,
        "source":      "video" | "sequence",
        "source_index": int,     # índice en la entrada original
        "clip_index":   int,     # 0 si es video corto o input sequence
    }
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from app.services.src.inference_helpers import (
    predict_sequence,
    resolve_flat_model,
)
from app.services.src.utils_pipeplanes import make_holistic
from app.services.video_processor import (
    DEFAULT_CLIP_SECONDS,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SHORT_THRESHOLD_S,
    video_to_sequences,
)


def predict(
    video_url: Optional[str] = None,
    video_urls: Optional[List[str]] = None,
    sequences: Optional[List[np.ndarray]] = None,
    clip_seconds: float       = DEFAULT_CLIP_SECONDS,
    short_threshold_s: float  = DEFAULT_SHORT_THRESHOLD_S,
    include_proba: bool       = False,
) -> List[dict]:
    """Predice con el modelo plano de 154 clases.

    Args:
        video_url:  un solo video (URL o path).
        video_urls: lista de videos.
        sequences:  lista de secuencias (45, 225) ya pre-procesadas.
        clip_seconds: duración del clip para videos largos.
        short_threshold_s: umbral por encima del cual se divide en clips.
        include_proba: añade ``proba`` al dict de cada predicción.

    Returns:
        list[dict] (puede tener 1 o más elementos). El orden refleja
        el orden de la entrada (y, dentro de cada video, el orden
        temporal de los clips).

    Raises:
        ValueError si no se pasa exactamente uno de los tres argumentos.
    """
    provided = sum(x is not None for x in (video_url, video_urls, sequences))
    if provided != 1:
        raise ValueError(
            "Debes pasar EXACTAMENTE uno de: video_url, video_urls o sequences."
        )

    model, info = resolve_flat_model()

    if sequences is not None:
        return _predict_sequences(
            model, info, sequences, include_proba=include_proba,
        )

    sources: List[str] = [video_url] if video_url is not None else list(video_urls)
    return _predict_videos(
        model, info, sources,
        clip_seconds=clip_seconds,
        short_threshold_s=short_threshold_s,
        include_proba=include_proba,
    )


# =====================================================================
# Helpers internos
# =====================================================================

def _predict_sequences(
    model,
    info,
    sequences: List[np.ndarray],
    include_proba: bool,
) -> List[dict]:
    """Predice cada secuencia pre-procesada. Una secuencia → una predicción."""
    out: List[dict] = []
    for src_i, seq in enumerate(sequences):
        result = predict_sequence(model, info, seq, include_proba=include_proba)
        result.update({
            "source":       "sequence",
            "source_index": src_i,
            "clip_index":   0,
        })
        out.append(result)
    return out


def _predict_videos(
    model,
    info,
    sources: List[str],
    clip_seconds: float,
    short_threshold_s: float,
    include_proba: bool,
) -> List[dict]:
    """Para cada video: lo divide en clips, extrae secuencias y predice.

    Reutiliza una sola instancia de MediaPipe Holistic entre todos los
    clips de todos los videos para amortizar el coste de inicialización.
    """
    out: List[dict] = []
    with make_holistic() as holistic:
        for src_i, source in enumerate(sources):
            try:
                seqs = video_to_sequences(
                    source,
                    sequence_length=info.sequence_length or DEFAULT_SEQUENCE_LENGTH,
                    clip_seconds=clip_seconds,
                    short_threshold_s=short_threshold_s,
                    holistic=holistic,
                )
            except Exception as exc:  # noqa: BLE001 - resiliente a fallos por video
                out.append({
                    "source":       "video",
                    "source_index": src_i,
                    "clip_index":   0,
                    "error":        f"{type(exc).__name__}: {exc}",
                    "video":        source,
                })
                continue

            if not seqs:
                out.append({
                    "source":       "video",
                    "source_index": src_i,
                    "clip_index":   0,
                    "error":        "no se pudieron extraer secuencias del video",
                    "video":        source,
                })
                continue

            for clip_i, seq in enumerate(seqs):
                result = predict_sequence(
                    model, info, seq, include_proba=include_proba,
                )
                result.update({
                    "source":       "video",
                    "source_index": src_i,
                    "clip_index":   clip_i,
                    "video":        source,
                })
                out.append(result)
    return out


__all__ = ["predict"]
