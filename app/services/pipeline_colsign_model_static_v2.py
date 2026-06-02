"""Servicio de inferencia del sub-modelo `colsign_lstm_norm_estatic_45_154_v2`.

Clasifica UNA seña dentro del grupo **'Grupo Estático'** (vocales/
consonantes estáticas del abecedario LSC, 21 clases).

Se invoca cuando el modelo raíz determinó que la seña pertenece al
grupo estático, o bien directamente desde un endpoint dedicado.

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
    resolve_sub_model,
)
from app.services.video_processor import video_to_single_sequence


GROUP_NAME = "Grupo Estático"


def predict_one(
    sequence: Optional[np.ndarray] = None,
    video_url: Optional[str] = None,
    video_path: Optional[str] = None,
    holistic = None,
    include_proba: bool = False,
) -> dict:
    """Predice la seña específica para una secuencia del Grupo Estático.

    Args:
        sequence:   ndarray (45, 225) keypoints normalizados.
        video_url:  URL pública del video corto.
        video_path: Path local del video corto.
        holistic:   MediaPipe Holistic reutilizable (opcional).
        include_proba: añade el vector completo de probas si True.

    Returns:
        dict con ``label`` (seña final), ``prob``, ``id``, ``model_name``,
        ``num_classes``.
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

    model, info = resolve_sub_model(GROUP_NAME)
    return predict_sequence(model, info, sequence, include_proba=include_proba)


__all__ = ["predict_one", "GROUP_NAME"]
