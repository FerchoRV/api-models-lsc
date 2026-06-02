"""Procesamiento de video para los pipelines ColSign.

Responsabilidades:

1. Abrir una fuente de video (URL pública HTTP/HTTPS o ruta local) usando
   OpenCV, que internamente delega en FFmpeg. Para URLs, esto streamea
   los bytes; es eficiente para videos cortos (los esperados, 2-10 s).
2. Calcular el FPS real del video y leer todos sus frames en memoria.
3. Dividir esos frames en clips de aproximadamente N segundos según el
   FPS. La regla es:
     - Video ≤ ``short_threshold_s`` (3 s por defecto)  → 1 solo clip.
     - Video >  ``short_threshold_s``                   → clips
       consecutivos no superpuestos de ``clip_seconds`` (2 s por default),
       descartando el residuo final si dura menos de ``min_clip_seconds``.
4. Convertir cada clip en una secuencia de keypoints ``(sequence_length,
   225)`` que los modelos LSTM esperan, reutilizando la lógica de
   ``utils_pipeplanes.extract_lstm_features`` para garantizar que el
   pre-procesamiento sea idéntico al usado durante entrenamiento.

Los pipelines (`pipeline_colsign_*`) consumen estas funciones; ellos NO
manipulan video directamente.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.services.src.utils_pipeplanes import (
    extract_lstm_features,
    make_holistic,
)


# Parámetros por defecto alineados con la arquitectura LSTM v2.
DEFAULT_SEQUENCE_LENGTH      = 45
DEFAULT_CLIP_SECONDS         = 2.0   # cada chunk dura ~2 s de video real
DEFAULT_SHORT_THRESHOLD_S    = 3.0   # ≤3 s → 1 sola secuencia
DEFAULT_MIN_CLIP_SECONDS     = 1.0   # residuos cortos se descartan
DEFAULT_FPS_FALLBACK         = 30.0  # si OpenCV no reporta FPS


# =====================================================================
# 1) Lectura de la fuente
# =====================================================================

def read_all_frames_with_fps(source: str) -> Tuple[List[np.ndarray], float]:
    """Lee TODOS los frames BGR de un video y devuelve ``(frames, fps)``.

    ``source`` puede ser:
        - una URL HTTP/HTTPS pública apuntando a un archivo de video
          (OpenCV usa FFmpeg para streamearla),
        - una ruta local.

    Si OpenCV no logra detectar FPS (cosa que pasa con algunos contenedores
    cuando se streamean), se devuelve ``0.0``; el caller decide el fallback.

    Raises:
        IOError: si la fuente no se puede abrir.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"No se pudo abrir el video: {source}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames, fps
    finally:
        cap.release()


# =====================================================================
# 2) División en clips de ~N segundos
# =====================================================================

def split_frames_into_clips(
    frames: List[np.ndarray],
    fps: float,
    clip_seconds: float        = DEFAULT_CLIP_SECONDS,
    short_threshold_s: float   = DEFAULT_SHORT_THRESHOLD_S,
    min_clip_seconds: float    = DEFAULT_MIN_CLIP_SECONDS,
    fps_fallback: float        = DEFAULT_FPS_FALLBACK,
) -> List[List[np.ndarray]]:
    """Divide la lista de frames en clips contiguos según el FPS.

    Política:
        - Si la duración total es ≤ ``short_threshold_s``, devuelve UN
          solo clip con todos los frames (típico para los videos de
          entrenamiento de 2-3 s).
        - Si es mayor, divide en clips consecutivos de
          ``round(fps * clip_seconds)`` frames cada uno. El último clip
          se conserva solo si dura ≥ ``min_clip_seconds`` (evita
          residuos demasiado cortos que producirían keypoints repetidos).
        - Si ``fps`` no es válido (0 o negativo), usa ``fps_fallback``.
    """
    if not frames:
        return []

    effective_fps = fps if fps and fps > 0 else fps_fallback
    total_seconds = len(frames) / effective_fps

    if total_seconds <= short_threshold_s:
        return [list(frames)]

    clip_size = max(1, int(round(effective_fps * clip_seconds)))
    min_size  = max(1, int(round(effective_fps * min_clip_seconds)))

    clips: List[List[np.ndarray]] = []
    for start in range(0, len(frames), clip_size):
        chunk = frames[start:start + clip_size]
        if len(chunk) >= min_size:
            clips.append(chunk)
    return clips


# =====================================================================
# 3) Clip → secuencia de keypoints
# =====================================================================

def clip_to_sequence(
    clip_frames: List[np.ndarray],
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    holistic = None,
) -> np.ndarray:
    """Convierte un clip (lista de frames BGR) en una matriz
    ``(sequence_length, 225)`` lista para alimentar a un LSTM v2.

    Internamente delega en ``utils_pipeplanes.extract_lstm_features`` para
    garantizar que el muestreo (np.linspace), la extracción Holistic y la
    normalización pose+manos sean IDÉNTICOS al pipeline de entrenamiento.

    Nota: deshabilitamos el trimming temporal (``trim_threshold_s=0``)
    porque ya estamos pasando un clip pre-recortado.
    """
    return extract_lstm_features(
        clip_frames,
        sequence_length=sequence_length,
        type_extract='pose_hands',
        normalize=True,
        drop_pose_visibility=True,
        holistic=holistic,
        trim_threshold_s=0.0,
        trim_tail_s=0.0,
    )


# =====================================================================
# 4) Atajos de alto nivel para los pipelines
# =====================================================================

def video_to_sequences(
    source: str,
    sequence_length: int      = DEFAULT_SEQUENCE_LENGTH,
    clip_seconds: float       = DEFAULT_CLIP_SECONDS,
    short_threshold_s: float  = DEFAULT_SHORT_THRESHOLD_S,
    min_clip_seconds: float   = DEFAULT_MIN_CLIP_SECONDS,
    holistic = None,
) -> List[np.ndarray]:
    """Procesa un video completo y devuelve N secuencias listas para
    predecir, donde N depende de la duración del video.

    - Si el video es corto (≤ ``short_threshold_s``), N=1.
    - Si es largo, N = cantidad de clips de ``clip_seconds`` que caben.

    Reutiliza una sola instancia de MediaPipe Holistic entre clips para
    evitar el coste de inicialización repetida.
    """
    frames, fps = read_all_frames_with_fps(source)
    clips = split_frames_into_clips(
        frames, fps,
        clip_seconds=clip_seconds,
        short_threshold_s=short_threshold_s,
        min_clip_seconds=min_clip_seconds,
    )
    if not clips:
        return []

    if holistic is not None:
        return [clip_to_sequence(c, sequence_length, holistic) for c in clips]

    with make_holistic() as h:
        return [clip_to_sequence(c, sequence_length, h) for c in clips]


def video_to_single_sequence(
    source: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    holistic = None,
) -> Optional[np.ndarray]:
    """Atajo para los sub-pipelines que SIEMPRE producen 1 predicción.

    Toma TODO el video como un único bloque y muestrea
    ``sequence_length`` frames uniformemente. Si el video es muy largo,
    esto comprime info; los sub-pipelines están pensados para clips
    cortos (2-3 s) que es lo que sus modelos fueron entrenados a ver.

    Devuelve ``None`` si no se pudieron leer frames.
    """
    frames, _fps = read_all_frames_with_fps(source)
    if not frames:
        return None
    return clip_to_sequence(frames, sequence_length, holistic)


__all__ = [
    "DEFAULT_SEQUENCE_LENGTH",
    "DEFAULT_CLIP_SECONDS",
    "DEFAULT_SHORT_THRESHOLD_S",
    "DEFAULT_MIN_CLIP_SECONDS",
    "read_all_frames_with_fps",
    "split_frames_into_clips",
    "clip_to_sequence",
    "video_to_sequences",
    "video_to_single_sequence",
]
