"""Helpers compartidos por los servicios `pipeline_colsign_*`.

Estas funciones encapsulan dos cosas que TODOS los pipelines necesitan:

1. **Resolver un modelo cargado**: primero busca en el diccionario global
   `ai_models` que llena `app/core/lifespan.py` durante el startup. Si por
   algún motivo el modelo no está allí (p.ej. test unitario, script suelto,
   o un sub-modelo opcional que falló al cargar), cae a `model_registry`
   y lo carga perezosamente. El caching de Keras lo hace `utils_pipeplanes`.

2. **Predecir desde una secuencia ya preparada**: corre el modelo sobre un
   ndarray ``(45, 225)`` y construye un dict de respuesta uniforme.

Cualquier endpoint que quiera servir un sub-modelo solo necesita
importar `predict_sequence` y el `resolve_*` correspondiente.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from app.core import model_registry
from app.core.config import settings
from app.core.lifespan import ai_models
from app.services.src.utils_pipeplanes import ModelInfo


# Pair (keras_model, ModelInfo). No anotamos keras.Model para no forzar
# el import de TensorFlow en este módulo.
ModelPair = Tuple[object, ModelInfo]

# Estos valores DEBEN coincidir con los que reportan los `*_labels.json`
# de la familia LSTM v2 (sequence_length=45, num_features=225 tras
# normalizar y drop_pose_visibility=True).
EXPECTED_SEQUENCE_LENGTH = 45
EXPECTED_NUM_FEATURES    = 225


# =====================================================================
# Resolución de modelos
# =====================================================================

def _hierarchical_slot() -> dict:
    """Devuelve `ai_models["colsign_jerarquico"]` o `{}` si aún no cargó."""
    return ai_models.get("colsign_jerarquico") or {}


def resolve_root_model() -> ModelPair:
    """Modelo raíz (4 clases de grupos morfológicos)."""
    pair = _hierarchical_slot().get("root")
    if pair is not None:
        return pair
    return model_registry.load_one(settings.COLSIGN_ROOT_MODEL)


def resolve_sub_model(group_name: str) -> ModelPair:
    """Sub-modelo asociado a un `group_name` (ej. 'Grupo Estático').

    El `group_name` debe ser EXACTAMENTE una de las claves de
    `settings.COLSIGN_SUB_MODELS` (que a su vez son las etiquetas que
    produce el modelo raíz).
    """
    pair = _hierarchical_slot().get("by_root", {}).get(group_name)
    if pair is not None:
        return pair

    model_name = settings.COLSIGN_SUB_MODELS.get(group_name)
    if model_name is None:
        raise KeyError(
            f"Grupo desconocido '{group_name}'. "
            f"Esperado uno de: {list(settings.COLSIGN_SUB_MODELS.keys())}"
        )
    return model_registry.load_one(model_name)


def resolve_flat_model() -> ModelPair:
    """Modelo plano (154 clases)."""
    pair = ai_models.get("colsign_plano")
    if pair is not None:
        return pair
    flat = model_registry.load_flat_model()
    if flat is None:
        raise FileNotFoundError(
            f"Modelo plano '{settings.COLSIGN_FLAT_MODEL}' no disponible."
        )
    return flat


# =====================================================================
# Validación y predicción
# =====================================================================

def validate_sequence(sequence: np.ndarray) -> np.ndarray:
    """Verifica el shape de una secuencia y la normaliza a float32.

    Devuelve la secuencia (posiblemente convertida) para usarla con
    seguridad en `model.predict`.
    """
    if not isinstance(sequence, np.ndarray):
        raise TypeError(
            f"sequence debe ser np.ndarray, llegó {type(sequence).__name__}"
        )
    if sequence.shape != (EXPECTED_SEQUENCE_LENGTH, EXPECTED_NUM_FEATURES):
        raise ValueError(
            f"Sequence shape inválido: {sequence.shape}, "
            f"esperado ({EXPECTED_SEQUENCE_LENGTH}, {EXPECTED_NUM_FEATURES})"
        )
    if sequence.dtype != np.float32:
        sequence = sequence.astype(np.float32, copy=False)
    return sequence


def predict_sequence(
    model,
    info: ModelInfo,
    sequence: np.ndarray,
    include_proba: bool = False,
) -> dict:
    """Predice top-1 sobre UNA secuencia preparada y devuelve un dict
    con un formato estándar para toda la API.

    Args:
        model: keras.Model ya cargado (resuelto por `resolve_*`).
        info:  ModelInfo paralelo a `model`.
        sequence: ndarray ``(45, 225)`` float32.
        include_proba: si True, agrega el vector completo de probas.

    Returns:
        {
            "label":       str,
            "prob":        float,
            "id":          int,
            "model_name":  str,
            "num_classes": int,
            "proba":       ndarray (num_classes,)   # solo si include_proba
        }
    """
    sequence = validate_sequence(sequence)
    proba = model.predict(sequence[None, ...], verbose=0)[0]
    proba = np.asarray(proba, dtype=np.float32)
    idx   = int(np.argmax(proba))

    result = {
        "label":       info.id_to_name.get(idx, f"<id={idx}>"),
        "prob":        float(proba[idx]),
        "id":          idx,
        "model_name":  info.name,
        "num_classes": info.num_classes,
    }
    if include_proba:
        result["proba"] = proba
    return result


__all__ = [
    "ModelPair",
    "EXPECTED_SEQUENCE_LENGTH",
    "EXPECTED_NUM_FEATURES",
    "resolve_root_model",
    "resolve_sub_model",
    "resolve_flat_model",
    "validate_sequence",
    "predict_sequence",
]
