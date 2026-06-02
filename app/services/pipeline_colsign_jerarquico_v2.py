"""Servicio de inferencia jerárquica (modelo raíz + 4 sub-modelos v2).

Compose los sub-pipelines ``pipeline_colsign_model_*`` para producir
la etiqueta final de una seña, siguiendo el flujo de dos pasos:

  1. El modelo raíz clasifica el grupo morfológico (4 clases).
  2. Según el grupo predicho, se enruta al sub-modelo correspondiente
     que produce la etiqueta concreta del lenguaje de señas.

Como todos los LSTM v2 comparten el mismo input shape ``(45, 225)``,
la secuencia de keypoints se extrae UNA sola vez por clip y se
reutiliza entre raíz y sub-modelo.

API
---
``predict(video_url=None, video_urls=None, sequences=None) -> list[dict]``

Solo UNO de los tres argumentos debe pasarse.

- ``video_url``  : URL pública (o path local) de UN video. Si dura ≤3s
                   genera 1 predicción; si dura más, se divide en
                   clips de 2s y devuelve N predicciones.
- ``video_urls`` : lista de URLs/paths. Resultados concatenados en orden.
- ``sequences``  : lista de ndarray ``(45, 225)`` pre-procesadas;
                   1 predicción por secuencia.

Cada elemento del resultado:
    {
        "label":             str,    # etiqueta FINAL (del sub-modelo)
        "prob":              float,
        "id":                int,
        "model_name":        str,    # nombre del sub-modelo usado
        "num_classes":       int,
        "grupo_raiz":        str,    # grupo predicho por el raíz
        "grupo_raiz_prob":   float,
        "grupo_raiz_model":  str,
        "source":            "video" | "sequence",
        "source_index":      int,
        "clip_index":        int,
    }

Si el sub-modelo correspondiente no está disponible, el dict tiene
``error`` en lugar de ``label`` final.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from app.core.config import settings
from app.services.pipeline_colsign_model_asimetrico import (
    predict_one as predict_asimetrico,
)
from app.services.pipeline_colsign_model_raiz_v2 import (
    predict_one as predict_raiz,
)
from app.services.pipeline_colsign_model_simetrico import (
    predict_one as predict_simetrico,
)
from app.services.pipeline_colsign_model_static_v2 import (
    predict_one as predict_static,
)
from app.services.pipeline_colsign_model_unimanual_v2 import (
    predict_one as predict_unimanual,
)
from app.services.src.inference_helpers import (
    EXPECTED_SEQUENCE_LENGTH,
    validate_sequence,
)
from app.services.src.utils_pipeplanes import make_holistic
from app.services.video_processor import (
    DEFAULT_CLIP_SECONDS,
    DEFAULT_SHORT_THRESHOLD_S,
    video_to_sequences,
)


# Mapeo grupo (etiqueta del modelo raíz) → función predict_one del sub-pipeline.
# Las claves DEBEN coincidir exactamente con `settings.COLSIGN_SUB_MODELS`
# (que a su vez son las etiquetas que produce el modelo raíz).
SUB_PIPELINES: Dict[str, Callable] = {
    "Grupo Estático":                     predict_static,
    "Grupo Dinámico Unimanual":           predict_unimanual,
    "Grupo Dinámico Bimanual Simétrico":  predict_simetrico,
    "Grupo Dinámico Bimanual Asimétrico": predict_asimetrico,
}


def predict(
    video_url: Optional[str] = None,
    video_urls: Optional[List[str]] = None,
    sequences: Optional[List[np.ndarray]] = None,
    clip_seconds: float       = DEFAULT_CLIP_SECONDS,
    short_threshold_s: float  = DEFAULT_SHORT_THRESHOLD_S,
    include_proba: bool       = False,
) -> List[dict]:
    """Predice con el pipeline jerárquico (raíz + sub-modelo).

    Args:
        video_url:   URL/path de un video.
        video_urls:  lista de URLs/paths.
        sequences:   lista de secuencias (45, 225) ya pre-procesadas.
        clip_seconds: duración del clip para videos largos.
        short_threshold_s: umbral por encima del cual se divide en clips.
        include_proba: añade ``proba`` del SUB-modelo al dict resultante.

    Returns:
        list[dict], 1+ elementos según la duración del video o el tamaño
        de ``sequences``.

    Raises:
        ValueError si no se pasa exactamente uno de los tres argumentos.
    """
    provided = sum(x is not None for x in (video_url, video_urls, sequences))
    if provided != 1:
        raise ValueError(
            "Debes pasar EXACTAMENTE uno de: video_url, video_urls o sequences."
        )

    # Sanity check: las claves del dispatcher cubren todos los grupos esperados.
    missing = set(settings.COLSIGN_SUB_MODELS) - set(SUB_PIPELINES)
    if missing:
        raise RuntimeError(
            f"SUB_PIPELINES no cubre estos grupos configurados: {missing}"
        )

    if sequences is not None:
        return _predict_from_sequences(
            sequences, include_proba=include_proba,
        )

    sources: List[str] = [video_url] if video_url is not None else list(video_urls)
    return _predict_from_videos(
        sources,
        clip_seconds=clip_seconds,
        short_threshold_s=short_threshold_s,
        include_proba=include_proba,
    )


# =====================================================================
# Núcleo del flujo jerárquico (1 secuencia → 1 predicción final)
# =====================================================================

def _route_sequence(
    sequence: np.ndarray,
    holistic = None,
    include_proba: bool = False,
) -> dict:
    """Aplica el pipeline jerárquico a UNA secuencia.

    1. Predice el grupo con el modelo raíz.
    2. Despacha al sub-pipeline asociado para obtener la etiqueta final.
    3. Compone un dict uniforme con info de ambos pasos.

    No envuelve errores: el caller (`_predict_from_*`) decide cómo
    presentar excepciones por entrada.
    """
    sequence = validate_sequence(sequence)

    root_pred = predict_raiz(sequence=sequence, holistic=holistic)
    group     = root_pred["label"]

    sub_fn = SUB_PIPELINES.get(group)
    if sub_fn is None:
        return {
            "label":            None,
            "prob":             None,
            "id":               None,
            "model_name":       None,
            "num_classes":      None,
            "grupo_raiz":       group,
            "grupo_raiz_prob":  root_pred["prob"],
            "grupo_raiz_model": root_pred["model_name"],
            "error": (
                f"El raíz predijo el grupo '{group}', "
                f"pero no hay sub-pipeline registrado para él."
            ),
        }

    sub_pred = sub_fn(
        sequence=sequence,
        holistic=holistic,
        include_proba=include_proba,
    )

    return {
        "label":            sub_pred["label"],
        "prob":             sub_pred["prob"],
        "id":               sub_pred["id"],
        "model_name":       sub_pred["model_name"],
        "num_classes":      sub_pred["num_classes"],
        "grupo_raiz":       group,
        "grupo_raiz_prob":  root_pred["prob"],
        "grupo_raiz_model": root_pred["model_name"],
        **({"proba": sub_pred["proba"]} if include_proba and "proba" in sub_pred else {}),
    }


# =====================================================================
# Predicción desde secuencias pre-procesadas
# =====================================================================

def _predict_from_sequences(
    sequences: List[np.ndarray],
    include_proba: bool,
) -> List[dict]:
    out: List[dict] = []
    for src_i, seq in enumerate(sequences):
        try:
            result = _route_sequence(seq, include_proba=include_proba)
        except Exception as exc:  # noqa: BLE001
            result = {
                "error": f"{type(exc).__name__}: {exc}",
            }
        result.update({
            "source":       "sequence",
            "source_index": src_i,
            "clip_index":   0,
        })
        out.append(result)
    return out


# =====================================================================
# Predicción desde videos (URL/path)
# =====================================================================

def _predict_from_videos(
    sources: List[str],
    clip_seconds: float,
    short_threshold_s: float,
    include_proba: bool,
) -> List[dict]:
    """Por cada video: extrae secuencias (1 o N) y enruta cada una.

    Reutiliza una sola instancia de MediaPipe Holistic entre todos los
    clips de todos los videos.
    """
    out: List[dict] = []
    with make_holistic() as holistic:
        for src_i, source in enumerate(sources):
            try:
                seqs = video_to_sequences(
                    source,
                    sequence_length=EXPECTED_SEQUENCE_LENGTH,
                    clip_seconds=clip_seconds,
                    short_threshold_s=short_threshold_s,
                    holistic=holistic,
                )
            except Exception as exc:  # noqa: BLE001
                out.append({
                    "source":       "video",
                    "source_index": src_i,
                    "clip_index":   0,
                    "video":        source,
                    "error":        f"{type(exc).__name__}: {exc}",
                })
                continue

            if not seqs:
                out.append({
                    "source":       "video",
                    "source_index": src_i,
                    "clip_index":   0,
                    "video":        source,
                    "error":        "no se pudieron extraer secuencias del video",
                })
                continue

            for clip_i, seq in enumerate(seqs):
                try:
                    result = _route_sequence(
                        seq, holistic=holistic, include_proba=include_proba,
                    )
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                result.update({
                    "source":       "video",
                    "source_index": src_i,
                    "clip_index":   clip_i,
                    "video":        source,
                })
                out.append(result)
    return out


__all__ = ["predict", "SUB_PIPELINES"]
