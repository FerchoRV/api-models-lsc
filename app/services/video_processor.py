"""Procesamiento de video para los pipelines ColSign.

Responsabilidades:

1. Abrir una fuente de video (URL pública HTTP/HTTPS o ruta local) usando
   OpenCV, que internamente delega en FFmpeg. Para URLs, esto streamea
   los bytes; es eficiente para videos cortos (los esperados, 2-10 s).
2. Calcular el FPS real del video y leer todos sus frames en memoria.
3. Dividir esos frames en clips de aproximadamente N segundos según un
   FPS efectivo. La regla es:
     - Video ≤ ``short_threshold_s`` (3 s por defecto)  → 1 solo clip.
     - Video >  ``short_threshold_s``                   → clips
       consecutivos no superpuestos de ``clip_seconds`` (2 s por default).
       El residuo final se conserva SOLO si tiene al menos
       ``min_frames`` frames (por defecto ``sequence_length`` = 45).
       Esto evita predecir sobre un clip residual demasiado corto, donde
       el muestreo a 45 keypoints terminaría repitiendo frames.
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
DEFAULT_FPS_FALLBACK         = 30.0  # si OpenCV no reporta FPS

# Tope absoluto de duración aceptada para evitar abuso / costes
# desmedidos. Videos más largos se rechazan ANTES de descargar todos
# los frames. El caller puede sobrescribirlo pasando `max_seconds`,
# o `0`/`None` para desactivar el límite.
DEFAULT_MAX_VIDEO_SECONDS    = 60.0


class VideoTooLongError(ValueError):
    """Levantada cuando un video excede la duración máxima permitida.

    Hereda de ``ValueError`` para que los handlers genéricos que ya
    capturan ``ValueError`` la traten como error de input. Los endpoints
    pueden capturarla específicamente para devolver HTTP 413 con
    información granular (``actual_seconds`` y ``max_seconds``).
    """

    def __init__(self, actual_seconds: float, max_seconds: float, source: str = ""):
        self.actual_seconds = float(actual_seconds)
        self.max_seconds    = float(max_seconds)
        self.source         = source
        super().__init__(
            f"El video dura {self.actual_seconds:.1f}s, "
            f"excede el máximo permitido de {self.max_seconds:.0f}s."
        )


# =====================================================================
# 1) Lectura de la fuente
# =====================================================================

def read_all_frames_with_fps(
    source: str,
    max_seconds: Optional[float] = DEFAULT_MAX_VIDEO_SECONDS,
) -> Tuple[List[np.ndarray], float]:
    """Lee TODOS los frames BGR de un video y devuelve ``(frames, fps)``.

    ``source`` puede ser:
        - una URL HTTP/HTTPS pública apuntando a un archivo de video
          (OpenCV usa FFmpeg para streamearla),
        - una ruta local.

    Si OpenCV no logra detectar FPS (cosa que pasa con algunos contenedores
    cuando se streamean), se devuelve ``0.0``; el caller decide el fallback.

    ``max_seconds`` (default ``DEFAULT_MAX_VIDEO_SECONDS`` = 60 s) limita
    la duración aceptada. El chequeo se hace en DOS pasos:

    1. **Pre-check por metadata** (rápido y barato): se consulta
       ``CAP_PROP_FRAME_COUNT`` y ``CAP_PROP_FPS`` ANTES de descargar
       todos los frames. Si los metadata son fiables y el video excede
       el límite, abortamos inmediatamente sin bajar bytes innecesarios.
    2. **Tope defensivo durante la lectura**: por si los metadata
       reportan ``frame_count = 0`` (algunos contenedores streaming),
       contamos los frames leídos y abortamos si excedemos
       ``max_seconds * effective_fps``. Esto protege la RAM contra
       videos que mintieron en sus metadata.

    Pasar ``max_seconds = 0`` o ``None`` desactiva ambos chequeos.

    Raises:
        IOError: si la fuente no se puede abrir.
        VideoTooLongError: si el video excede ``max_seconds``.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"No se pudo abrir el video: {source}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        check_enabled = bool(max_seconds and max_seconds > 0)

        # 1) Pre-check por metadata. Usamos el FPS efectivo (al menos 30)
        # para no sub-estimar duración cuando OpenCV reporta FPS bajo.
        if check_enabled:
            effective_fps = _resolve_effective_fps(fps)
            reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if reported_count > 0:
                estimated_seconds = reported_count / effective_fps
                if estimated_seconds > max_seconds:
                    raise VideoTooLongError(
                        actual_seconds=estimated_seconds,
                        max_seconds=max_seconds,
                        source=source,
                    )

        # 2) Lectura completa con tope defensivo.
        # `+1` da margen para que el límite se detecte en cuanto se
        # SUPERA, no cuando se IGUALA exacto.
        max_frames = (
            int(_resolve_effective_fps(fps) * max_seconds) + 1
            if check_enabled
            else None
        )
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if max_frames is not None and len(frames) > max_frames:
                raise VideoTooLongError(
                    actual_seconds=len(frames) / _resolve_effective_fps(fps),
                    max_seconds=max_seconds,
                    source=source,
                )
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
    fps_fallback: float        = DEFAULT_FPS_FALLBACK,
    min_frames: int            = DEFAULT_SEQUENCE_LENGTH,
) -> List[List[np.ndarray]]:
    """Divide la lista de frames en clips contiguos según el FPS.

    Política:
        - Si la duración total es ≤ ``short_threshold_s``, devuelve UN
          solo clip con todos los frames (típico para los videos de
          entrenamiento de 2-3 s).
        - Si es mayor, divide en clips consecutivos de
          ``round(effective_fps * clip_seconds)`` frames cada uno.
          Los rangos NO se comparten: si el clip size es 60, los cortes
          son [0:60], [60:120], [120:180], etc.
        - ``effective_fps`` usa al menos ``fps_fallback``. Esto evita
          sobre-dividir videos de Firebase/URLs públicas cuando OpenCV
          reporta un FPS menor al real (por ejemplo 24 en un video que
          fue grabado/exportado a ~30 FPS).
        - Los clips con MENOS de ``min_frames`` frames se descartan.
          Por defecto ``min_frames`` = ``DEFAULT_SEQUENCE_LENGTH`` (45),
          porque debajo de eso el muestreo a 45 keypoints tendría que
          repetir frames y la predicción no sería confiable. Esto solo
          aplica al residuo final: los clips internos siempre tienen
          exactamente ``clip_size`` frames.

    Ejemplos (effective_fps=30, clip_seconds=2 → clip_size=60, min_frames=45):
        - 200 frames → [0:60], [60:120], [120:180]; residuo [180:200]
          (20 frames) se descarta por ser < 45.
        - 220 frames → [0:60], [60:120], [120:180], [180:220]; el último
          tiene 40 frames y también se descarta → quedan 3 clips.
        - 250 frames → [0:60], [60:120], [120:180], [180:240]; residuo
          [240:250] se descarta → quedan 4 clips.
    """
    if not frames:
        return []

    effective_fps = _resolve_effective_fps(fps, fps_fallback)
    total_seconds = len(frames) / effective_fps

    if total_seconds <= short_threshold_s:
        # Si el video es corto pero tiene MENOS de min_frames, igual
        # devolvemos el único clip: extract_lstm_features hace muestreo
        # con repetición y el caller (un sub-pipeline) ya asume que
        # quiere SIEMPRE una predicción, aún con poca info.
        return [list(frames)]

    clip_size = max(1, int(round(effective_fps * clip_seconds)))
    min_size  = max(1, int(min_frames))

    clips: List[List[np.ndarray]] = []
    for start in range(0, len(frames), clip_size):
        chunk = frames[start:start + clip_size]
        if len(chunk) >= min_size:
            clips.append(chunk)
    return clips


def _resolve_effective_fps(
    reported_fps: float,
    fps_fallback: float = DEFAULT_FPS_FALLBACK,
) -> float:
    """Normaliza el FPS usado para cortar clips.

    OpenCV puede reportar FPS bajos o inconsistentes al leer videos desde
    URLs públicas. En este proyecto los videos de entrenamiento y prueba
    se manejan alrededor de 30 FPS; si el metadata reporta menos, usarlo
    provoca más clips de los esperados (p.ej. un video de ~120 frames se
    corta en 3 partes con 24 FPS en vez de 2 partes con 30 FPS).

    Por eso usamos `max(reported_fps, fps_fallback)` cuando hay FPS
    válido, y `fps_fallback` cuando no lo hay.
    """
    if not reported_fps or reported_fps <= 0:
        return fps_fallback
    return max(float(reported_fps), float(fps_fallback))


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
    sequence_length: int        = DEFAULT_SEQUENCE_LENGTH,
    clip_seconds: float         = DEFAULT_CLIP_SECONDS,
    short_threshold_s: float    = DEFAULT_SHORT_THRESHOLD_S,
    max_seconds: Optional[float] = DEFAULT_MAX_VIDEO_SECONDS,
    holistic = None,
) -> List[np.ndarray]:
    """Procesa un video completo y devuelve N secuencias listas para
    predecir, donde N depende de la duración del video.

    - Si el video es corto (≤ ``short_threshold_s``), N=1.
    - Si es largo, N = cantidad de clips de ``clip_seconds`` que caben,
      descartando residuos finales con menos de ``sequence_length``
      frames (porque no alcanzarían para un muestreo limpio a 45).

    ``max_seconds`` impone un tope de duración (default 60 s); pásalo
    en ``0`` o ``None`` para desactivar la validación.

    Reutiliza una sola instancia de MediaPipe Holistic entre clips para
    evitar el coste de inicialización repetida.
    """
    frames, fps = read_all_frames_with_fps(source, max_seconds=max_seconds)
    clips = split_frames_into_clips(
        frames, fps,
        clip_seconds=clip_seconds,
        short_threshold_s=short_threshold_s,
        min_frames=sequence_length,
    )
    if not clips:
        return []

    if holistic is not None:
        return [clip_to_sequence(c, sequence_length, holistic) for c in clips]

    with make_holistic() as h:
        return [clip_to_sequence(c, sequence_length, h) for c in clips]


def video_to_single_sequence(
    source: str,
    sequence_length: int          = DEFAULT_SEQUENCE_LENGTH,
    max_seconds: Optional[float]  = DEFAULT_MAX_VIDEO_SECONDS,
    holistic = None,
) -> Optional[np.ndarray]:
    """Atajo para los sub-pipelines que SIEMPRE producen 1 predicción.

    Toma TODO el video como un único bloque y muestrea
    ``sequence_length`` frames uniformemente. Si el video es muy largo,
    esto comprime info; los sub-pipelines están pensados para clips
    cortos (2-3 s) que es lo que sus modelos fueron entrenados a ver.

    ``max_seconds`` impone un tope de duración (default 60 s); pásalo
    en ``0`` o ``None`` para desactivar la validación.

    Devuelve ``None`` si no se pudieron leer frames.
    """
    frames, _fps = read_all_frames_with_fps(source, max_seconds=max_seconds)
    if not frames:
        return None
    return clip_to_sequence(frames, sequence_length, holistic)


__all__ = [
    "DEFAULT_SEQUENCE_LENGTH",
    "DEFAULT_CLIP_SECONDS",
    "DEFAULT_SHORT_THRESHOLD_S",
    "DEFAULT_MAX_VIDEO_SECONDS",
    "VideoTooLongError",
    "read_all_frames_with_fps",
    "split_frames_into_clips",
    "clip_to_sequence",
    "video_to_sequences",
    "video_to_single_sequence",
]
