"""Servicio de inferencia INDIVIDUAL del modelo plano `colsign_lstm_norm_45_154`.

A diferencia de `pipeline_colsign_45_154.py` (que es batch y puede
devolver N predicciones por video largo), este módulo está pensado
para predicciones individuales: SIEMPRE devuelve UNA sola predicción,
sin importar la duración del video.

Si el video dura más de los 2-3 s esperados, se muestrean
``sequence_length`` (45) frames uniformemente del video completo y se
predice con esa única secuencia.

API
---
``predict_one(sequence=None, video_url=None, video_path=None, holistic=None) -> dict``
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.services.src.inference_helpers import (
    EXPECTED_SEQUENCE_LENGTH,
    predict_sequence,
    resolve_flat_model,
)
from app.services.video_processor import video_to_single_sequence


def predict_one(
    sequence: Optional[np.ndarray] = None,
    video_url: Optional[str] = None,
    video_path: Optional[str] = None,
    holistic = None,
    include_proba: bool = False,
) -> dict:
    """Predice UNA seña con el modelo plano de 154 clases.

    Args:
        sequence:   ndarray (45, 225) keypoints normalizados.
        video_url:  URL pública del video (cualquier duración).
        video_path: Path local del video (cualquier duración).
        holistic:   MediaPipe Holistic reutilizable (opcional).
        include_proba: añade el vector completo de probas si True.

    Returns:
        dict con ``label`` (seña), ``prob``, ``id``, ``model_name``,
        ``num_classes``.

    Raises:
        ValueError si no se pasa exactamente una fuente o si la
        secuencia tiene shape inválido.
    """
    provided = sum(x is not None for x in (sequence, video_url, video_path))
    if provided != 1:
        raise ValueError(
            "Debes pasar EXACTAMENTE uno de: sequence, video_url o video_path."
        )

    if sequence is None:
        source = video_url if video_url is not None else video_path
        sequence = video_to_single_sequence(
            source,
            sequence_length=EXPECTED_SEQUENCE_LENGTH,
            holistic=holistic,
        )
        if sequence is None:
            raise ValueError(
                f"No se pudieron extraer keypoints del video: {source}"
            )

    model, info = resolve_flat_model()
    return predict_sequence(model, info, sequence, include_proba=include_proba)


__all__ = ["predict_one"]
