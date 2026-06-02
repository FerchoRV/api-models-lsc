"""Servicio de inferencia del modelo raíz `colsign_lstm_norm_raiz_45_154_v2`.

Clasifica el GRUPO morfológico de UNA seña (4 clases):
    - Grupo Estático
    - Grupo Dinámico Unimanual
    - Grupo Dinámico Bimanual Simétrico
    - Grupo Dinámico Bimanual Asimétrico

Este pipeline NO produce la etiqueta final del lenguaje de señas: para
eso el pipeline jerárquico encadena este modelo con el sub-modelo del
grupo predicho. Cuando se monte un endpoint que sirva SOLO el raíz, se
usará `predict_one` directamente.

API
---
``predict_one(sequence=None, video_url=None, video_path=None, holistic=None) -> dict``

Solo uno de ``sequence`` / ``video_url`` / ``video_path`` debe pasarse.
Siempre devuelve UNA predicción.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.services.src.inference_helpers import (
    EXPECTED_SEQUENCE_LENGTH,
    predict_sequence,
    resolve_root_model,
)
from app.services.video_processor import video_to_single_sequence


def predict_one(
    sequence: Optional[np.ndarray] = None,
    video_url: Optional[str] = None,
    video_path: Optional[str] = None,
    holistic = None,
    include_proba: bool = False,
) -> dict:
    """Predice el grupo morfológico de UNA seña.

    Args:
        sequence:   ndarray (45, 225) con keypoints ya normalizados.
        video_url:  URL pública del video corto (alternativa).
        video_path: Path local del video corto (alternativa).
        holistic:   instancia MediaPipe Holistic reutilizable (opcional).
        include_proba: si True, agrega el vector completo de probabilidades.

    Returns:
        dict con ``label`` (grupo), ``prob``, ``id``, ``model_name``,
        ``num_classes``. Si ``include_proba``, también ``proba``.

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

    model, info = resolve_root_model()
    return predict_sequence(model, info, sequence, include_proba=include_proba)


__all__ = ["predict_one"]
