"""Procesamiento de video por VENTANA DESLIZANTE para los pipelines ColSign.

Diferencia con `video_processor.py`
-----------------------------------
`video_processor.py` corta el video en clips contiguos y NO superpuestos
(``[0:60], [60:120], ...``). Eso funciona bien cuando el cliente graba un
clip por seña, pero falla con video continuo (varias señas seguidas):
una seña puede quedar partida justo en el borde de dos clips y perderse.

Este módulo implementa una estrategia distinta, pensada para video
continuo:

1. **Ventana de tiempo deslizante (sliding window).** Una ventana de
   ``window_seconds`` recorre el video avanzando ``stride_seconds`` en
   cada paso. Como ``stride < window``, las ventanas se SOLAPAN, de modo
   que cualquier seña cae completa dentro de al menos una ventana.

2. **Muestreo idéntico al entrenamiento.** Cada ventana se reduce a una
   secuencia ``(sequence_length, 225)`` con el MISMO muestreo
   (``compute_target_indices`` = ``np.linspace``) y la MISMA normalización
   (``normalize_pose_hands_keypoints``) que usa `extract_lstm_features`.
   Para no re-ejecutar MediaPipe sobre los mismos frames una y otra vez
   (las ventanas se solapan), los keypoints crudos se extraen UNA sola
   vez por frame y luego cada ventana solo muestrea de esa caché.

3. **Filtro por umbral de confianza.** Las predicciones de ventana con
   ``prob < min_confidence`` se descartan: son típicamente transiciones
   entre señas donde el modelo "adivina".

4. **Penalización de repeticiones (supresión temporal).** Como la misma
   seña cae en varias ventanas solapadas, se generan detecciones
   repetidas. Se colapsan en UN solo evento: mientras la misma etiqueta
   reaparezca dentro de ``repeat_gap_seconds``, se considera la misma
   seña y se conserva la detección de MAYOR confianza. Si la etiqueta
   vuelve a aparecer después de un hueco mayor a ``repeat_gap_seconds``,
   se trata como una ocurrencia NUEVA (p.ej. una seña repetida a
   propósito).

5. **Descarte de cola.** Los últimos ``discard_tail_seconds`` del video
   suelen ser ruido (la persona baja las manos / apaga la cámara), así
   que se ignoran antes de ventanear.

El resultado es una lista de eventos en orden temporal, directamente
consumible por los endpoints narrativos de `translation.py` (cada evento
trae ``label`` y ``prob``, que es lo que `_build_narrative` necesita).

Parámetros por defecto (overridables vía API)
---------------------------------------------
    SEQUENCE_LENGTH      = 45
    WINDOW_SECONDS       = 2.0
    STRIDE_SECONDS       = 1.0
    MIN_CONFIDENCE       = 0.50
    REPEAT_GAP_SECONDS   = 3.0
    DISCARD_TAIL_SECONDS = 2.0
    MIN_SEGMENT_FRAMES   = 10
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from app.services.src.utils import (
    compute_target_indices,
    extract_keypoints_pose_hands,
    mediapipe_detection,
    normalize_pose_hands_keypoints,
)
from app.services.src.utils_pipeplanes import make_holistic
from app.services.video_processor import (
    DEFAULT_FPS_FALLBACK,
    DEFAULT_MAX_VIDEO_SECONDS,
    VideoTooLongError,
    read_all_frames_with_fps,
)


# =====================================================================
# Parámetros por defecto de la ventana deslizante
# =====================================================================
SEQUENCE_LENGTH      = 45
WINDOW_SECONDS       = 2.0
STRIDE_SECONDS       = 1.0
MIN_CONFIDENCE       = 0.50
REPEAT_GAP_SECONDS   = 3.0
DISCARD_TAIL_SECONDS = 2.0
MIN_SEGMENT_FRAMES   = 10

# Tamaño del vector crudo pose+manos antes de normalizar (33*4 + 21*3 + 21*3).
_RAW_FEATURES = 258

# Banda de FPS plausible para una grabación real. Fuera de esta banda
# asumimos que OpenCV/FFmpeg reportó basura (típico al streamear URLs:
# devuelven el timebase del contenedor, p.ej. 600, 1000 o 90000) y caemos
# al fallback de 30. A diferencia de los cortes exactos en
# `video_processor.py`, aquí NO se usa `max(reported, 30)`: ese "piso"
# infla el tamaño de ventana cuando el FPS reportado es absurdamente alto
# y haría que toda la grabación quede en UNA sola ventana.
_WINDOW_FPS_MIN = 5.0
_WINDOW_FPS_MAX = 120.0


def _resolve_windowing_fps(reported_fps: float) -> float:
    """FPS confiable para mapear segundos→frames en la ventana deslizante.

    Confía en ``reported_fps`` solo si cae dentro de una banda plausible
    (``[5, 120]``); en cualquier otro caso (0, negativo, minúsculo o
    gigante) usa ``DEFAULT_FPS_FALLBACK`` (30).
    """
    if reported_fps and _WINDOW_FPS_MIN <= reported_fps <= _WINDOW_FPS_MAX:
        return float(reported_fps)
    return float(DEFAULT_FPS_FALLBACK)


# Una función de predicción recibe una secuencia (45, 225) y devuelve un
# dict con al menos {"label": str, "prob": float}. La proveen los
# pipelines (plano o jerárquico) vía las funciones `predict_*` de abajo.
PredictFn = Callable[[np.ndarray], dict]


# =====================================================================
# 1) Extracción de keypoints crudos por frame (una sola pasada)
# =====================================================================

def _extract_all_frame_keypoints(frames: List[np.ndarray], holistic) -> np.ndarray:
    """Ejecuta MediaPipe Holistic sobre CADA frame y devuelve ``(N, 258)``.

    Se hace una sola vez por video. Las ventanas posteriores solo muestrean
    filas de esta matriz, evitando reprocesar los frames solapados.
    """
    raw = np.empty((len(frames), _RAW_FEATURES), dtype=np.float32)
    for i, frame in enumerate(frames):
        _, results = mediapipe_detection(frame, holistic)
        raw[i] = extract_keypoints_pose_hands(results)
    return raw


def _window_to_sequence(
    raw_all: np.ndarray,
    start: int,
    end: int,
    sequence_length: int,
) -> np.ndarray:
    """Convierte el tramo ``[start:end)`` de keypoints crudos en una
    secuencia normalizada ``(sequence_length, 225)``.

    Replica exactamente lo que hace `extract_lstm_features`: muestreo
    uniforme con `compute_target_indices` + normalización pose/manos.
    """
    n = end - start
    local_idx = compute_target_indices(n, sequence_length)  # 0..n-1
    abs_idx = local_idx + start
    raw = raw_all[abs_idx]                                   # (45, 258)
    return normalize_pose_hands_keypoints(raw, drop_pose_visibility=True)


# =====================================================================
# 2) Penalización de repeticiones (supresión temporal por etiqueta)
# =====================================================================

def _suppress_repeats(
    detections: List[dict],
    repeat_gap_seconds: float,
) -> List[dict]:
    """Colapsa detecciones repetidas de la misma etiqueta en eventos únicos.

    ``detections`` debe venir ordenado por tiempo. Cada detección lleva la
    predicción completa más ``_t_start``, ``_t_end`` y ``_t_center`` (s).

    Regla: mientras una etiqueta reaparezca a ≤ ``repeat_gap_seconds`` del
    fin de su última aparición, es la MISMA seña → se fusiona y se conserva
    la detección de mayor ``prob``. Si reaparece tras un hueco mayor, es
    una ocurrencia nueva.
    """
    events: List[dict] = []
    active: Dict[str, dict] = {}  # label → evento más reciente (mutable, dentro de events)

    for det in detections:
        label = det["label"]
        ev = active.get(label)
        gap = None if ev is None else det["_t_start"] - ev["t_end"]

        if ev is not None and gap is not None and gap <= repeat_gap_seconds:
            # Misma seña en curso: extiende el evento y penaliza la repetición.
            ev["t_end"] = max(ev["t_end"], det["_t_end"])
            ev["repeat_count"] += 1
            if det["prob"] > ev["prob"]:
                # Nos quedamos con la detección más confiable, pero
                # preservamos las marcas temporales agregadas.
                t_start = ev["t_start"]
                t_end = ev["t_end"]
                rc = ev["repeat_count"]
                ev.update(_event_payload(det))
                ev["t_start"] = t_start
                ev["t_end"] = t_end
                ev["repeat_count"] = rc
                ev["t_peak"] = det["_t_center"]
        else:
            # Nueva ocurrencia.
            new_ev = _event_payload(det)
            new_ev["t_start"] = det["_t_start"]
            new_ev["t_end"] = det["_t_end"]
            new_ev["t_peak"] = det["_t_center"]
            new_ev["repeat_count"] = 1
            events.append(new_ev)
            active[label] = new_ev

    events.sort(key=lambda e: e["t_start"])
    for e in events:
        e["t_start"] = round(e["t_start"], 3)
        e["t_end"] = round(e["t_end"], 3)
        e["t_peak"] = round(e["t_peak"], 3)
    return events


def _event_payload(det: dict) -> dict:
    """Copia la predicción de una detección sin sus campos internos ``_t_*``."""
    return {k: v for k, v in det.items() if not k.startswith("_")}


# =====================================================================
# 3) Núcleo: recorrer un video con ventana deslizante
# =====================================================================

def _predict_one_source(
    source: str,
    source_index: int,
    predict_fn: PredictFn,
    holistic,
    *,
    sequence_length: int,
    window_seconds: float,
    stride_seconds: float,
    min_confidence: float,
    repeat_gap_seconds: float,
    discard_tail_seconds: float,
    min_segment_frames: int,
    max_seconds: Optional[float],
    include_proba: bool,
) -> List[dict]:
    """Procesa UN video con ventana deslizante y devuelve eventos únicos."""
    frames, fps = read_all_frames_with_fps(source, max_seconds=max_seconds)
    if not frames:
        return [{
            "source": "video",
            "source_index": source_index,
            "video": source,
            "error": "no se pudieron leer frames del video",
        }]

    effective_fps = _resolve_windowing_fps(fps)

    # Descarte de cola: ignora los últimos `discard_tail_seconds` si queda
    # suficiente video para al menos un segmento válido.
    n_total = len(frames)
    tail = int(round(discard_tail_seconds * effective_fps))
    if tail > 0 and (n_total - tail) >= min_segment_frames:
        frames = frames[: n_total - tail]
    n = len(frames)

    raw_all = _extract_all_frame_keypoints(frames, holistic)

    window_size = max(1, int(round(effective_fps * window_seconds)))
    stride = max(1, int(round(effective_fps * stride_seconds)))

    # Guard defensivo: si la ventana abarca (casi) todo el video pero hay
    # frames de sobra para varias ventanas, el FPS sigue siendo poco fiable.
    # Recalculamos asumiendo el fallback de 30 para garantizar solapamiento
    # y múltiples ventanas en grabaciones largas.
    if window_size >= n and n >= 2 * min_segment_frames:
        effective_fps = float(DEFAULT_FPS_FALLBACK)
        window_size = max(1, int(round(effective_fps * window_seconds)))
        stride = max(1, int(round(effective_fps * stride_seconds)))

    # La ventana nunca debe superar el total de frames disponibles.
    window_size = min(window_size, n)
    # El stride no puede igualar/superar la ventana (perderíamos el solape).
    stride = max(1, min(stride, window_size))

    detections: List[dict] = []
    seen_end_at_n = False
    for start in range(0, n, stride):
        end = min(start + window_size, n)
        seg_len = end - start
        if seg_len < min_segment_frames:
            continue
        if end == n:
            seen_end_at_n = True

        seq = _window_to_sequence(raw_all, start, end, sequence_length)
        pred = predict_fn(seq)

        if pred.get("label") is None or pred.get("prob") is None:
            continue
        if float(pred["prob"]) < min_confidence:
            continue

        det = dict(pred)
        det["_t_start"] = start / effective_fps
        det["_t_end"] = end / effective_fps
        det["_t_center"] = (start + end) / 2.0 / effective_fps
        detections.append(det)

    # Fallback para videos cortos: si la ventana no llegó hasta el final
    # (o no produjo nada) pero hay suficientes frames, evalúa una última
    # ventana que cubra el tramo final completo.
    if n >= min_segment_frames and not seen_end_at_n:
        start = max(0, n - window_size)
        seq = _window_to_sequence(raw_all, start, n, sequence_length)
        pred = predict_fn(seq)
        if pred.get("label") is not None and pred.get("prob") is not None \
                and float(pred["prob"]) >= min_confidence:
            det = dict(pred)
            det["_t_start"] = start / effective_fps
            det["_t_end"] = n / effective_fps
            det["_t_center"] = (start + n) / 2.0 / effective_fps
            detections.append(det)

    events = _suppress_repeats(detections, repeat_gap_seconds)

    for ev in events:
        ev["source"] = "video"
        ev["source_index"] = source_index
        ev["video"] = source
        if not include_proba:
            ev.pop("proba", None)
    return events


def predict_sliding_window(
    predict_fn: PredictFn,
    video_url: Optional[str] = None,
    video_urls: Optional[List[str]] = None,
    *,
    sequence_length: int        = SEQUENCE_LENGTH,
    window_seconds: float       = WINDOW_SECONDS,
    stride_seconds: float       = STRIDE_SECONDS,
    min_confidence: float       = MIN_CONFIDENCE,
    repeat_gap_seconds: float   = REPEAT_GAP_SECONDS,
    discard_tail_seconds: float = DISCARD_TAIL_SECONDS,
    min_segment_frames: int     = MIN_SEGMENT_FRAMES,
    max_seconds: Optional[float] = DEFAULT_MAX_VIDEO_SECONDS,
    include_proba: bool         = False,
) -> List[dict]:
    """Aplica ventana deslizante + penalización a uno o varios videos.

    Args:
        predict_fn: función que mapea una secuencia ``(45, 225)`` a un dict
            de predicción (``{"label", "prob", ...}``). La proveen
            `predict_flat` / `predict_hierarchical`.
        video_url / video_urls: fuente(s). Debe pasarse EXACTAMENTE una.
        sequence_length: T de cada secuencia (default 45).
        window_seconds: ancho de la ventana en segundos.
        stride_seconds: avance entre ventanas (menor que window → solape).
        min_confidence: umbral mínimo de ``prob`` para conservar una ventana.
        repeat_gap_seconds: hueco máximo para considerar dos detecciones de
            la misma etiqueta como la misma seña.
        discard_tail_seconds: segundos finales del video que se ignoran.
        min_segment_frames: tamaño mínimo (en frames) de una ventana válida.
        max_seconds: tope de duración del video (default 60 s).
        include_proba: conserva el vector ``proba`` en cada evento.

    Returns:
        list[dict] de eventos en orden temporal. Cada evento incluye
        ``label``, ``prob``, ``t_start``, ``t_end``, ``t_peak`` y
        ``repeat_count``, además de la metadata del modelo.

    Raises:
        ValueError si no se pasa exactamente una fuente.
    """
    provided = sum(x is not None for x in (video_url, video_urls))
    if provided != 1:
        raise ValueError("Debes pasar EXACTAMENTE uno de: video_url o video_urls.")

    sources: List[str] = [video_url] if video_url is not None else list(video_urls)

    out: List[dict] = []
    with make_holistic() as holistic:
        for src_i, source in enumerate(sources):
            try:
                out.extend(_predict_one_source(
                    source, src_i, predict_fn, holistic,
                    sequence_length=sequence_length,
                    window_seconds=window_seconds,
                    stride_seconds=stride_seconds,
                    min_confidence=min_confidence,
                    repeat_gap_seconds=repeat_gap_seconds,
                    discard_tail_seconds=discard_tail_seconds,
                    min_segment_frames=min_segment_frames,
                    max_seconds=max_seconds,
                    include_proba=include_proba,
                ))
            except VideoTooLongError:
                # Se propaga para que el endpoint devuelva HTTP 413.
                raise
            except Exception as exc:  # noqa: BLE001 - un video roto no tumba el resto
                out.append({
                    "source":       "video",
                    "source_index": src_i,
                    "video":        source,
                    "error":        f"{type(exc).__name__}: {exc}",
                })
    return out


# =====================================================================
# 4) Entrypoints por modelo (construyen el predict_fn adecuado)
# =====================================================================

def predict_flat(
    video_url: Optional[str] = None,
    video_urls: Optional[List[str]] = None,
    **kwargs,
) -> List[dict]:
    """Ventana deslizante usando el modelo PLANO de 154 clases."""
    from app.services.src.inference_helpers import (
        predict_sequence,
        resolve_flat_model,
    )

    model, info = resolve_flat_model()

    def _predict_fn(seq: np.ndarray) -> dict:
        return predict_sequence(model, info, seq)

    return predict_sliding_window(
        _predict_fn, video_url=video_url, video_urls=video_urls, **kwargs,
    )


def predict_hierarchical(
    video_url: Optional[str] = None,
    video_urls: Optional[List[str]] = None,
    **kwargs,
) -> List[dict]:
    """Ventana deslizante usando el pipeline JERÁRQUICO (raíz + sub-modelo)."""
    from app.services.pipeline_colsign_jerarquico_v2 import (
        predict_sequence_hierarchical,
    )

    def _predict_fn(seq: np.ndarray) -> dict:
        return predict_sequence_hierarchical(seq, holistic=None)

    return predict_sliding_window(
        _predict_fn, video_url=video_url, video_urls=video_urls, **kwargs,
    )


__all__ = [
    "SEQUENCE_LENGTH",
    "WINDOW_SECONDS",
    "STRIDE_SECONDS",
    "MIN_CONFIDENCE",
    "REPEAT_GAP_SECONDS",
    "DISCARD_TAIL_SECONDS",
    "MIN_SEGMENT_FRAMES",
    "predict_sliding_window",
    "predict_flat",
    "predict_hierarchical",
]
