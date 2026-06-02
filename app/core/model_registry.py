"""Registro central de modelos ColSign para la API.

Este módulo es la única capa que la API debería usar para obtener
modelos cargados. Encapsula:

- Las rutas absolutas a `app/models/` y `app/info_models/` (vía `settings`).
- Los nombres canónicos del modelo raíz, los sub-modelos y el plano.
- La lógica real de descubrimiento, resolución de `_best.keras` y carga
  Keras, que vive en `app/services/src/utils_pipeplanes.py`.

Patrón de uso desde el lifespan o desde un endpoint:

    from app.core import model_registry

    hierarchical = model_registry.load_hierarchical_set()
    root_model, root_info = hierarchical["root"]
    sub_model, sub_info   = hierarchical["by_root"]["Grupo Estático"]

    flat_model, flat_info = model_registry.load_flat_model()

`utils_pipeplanes.load_model` ya cachea por ruta `.keras`, así que
llamar a estas funciones varias veces NO recarga los pesos.
"""

from __future__ import annotations

from typing import Optional, Tuple

from app.core.config import settings
from app.services.src.utils_pipeplanes import (
    ModelInfo,
    load_model as _load_model,
)


# Tipo de retorno común: (keras_model, ModelInfo). No anotamos el
# keras_model con `tf.keras.Model` para no forzar el import de TF aquí
# (queremos que TF se cargue de forma perezosa).
ModelPair = Tuple[object, ModelInfo]


def load_one(model_name: str) -> ModelPair:
    """Carga un modelo por su nombre canónico (sin sufijo `_best`).

    Usa las rutas absolutas configuradas en `settings`, por lo que es
    seguro llamarla desde cualquier proceso/cwd (uvicorn, scripts, etc.).
    """
    return _load_model(
        model_name,
        models_dir=settings.MODELS_DIR,
        info_dir=settings.INFO_MODELS_DIR,
    )


def load_hierarchical_set() -> dict:
    """Carga el conjunto completo del pipeline jerárquico v2.

    Returns:
        {
            "root":    (keras_model, ModelInfo),  # 4 clases (grupos)
            "by_root": {
                "Grupo Estático":                     (model, info) | None,
                "Grupo Dinámico Unimanual":           (model, info) | None,
                "Grupo Dinámico Bimanual Simétrico":  (model, info) | None,
                "Grupo Dinámico Bimanual Asimétrico": (model, info) | None,
            },
        }

    Si algún sub-modelo no se puede cargar (p.ej. falta su `.keras`),
    se deja como `None` en lugar de abortar todo el set: el caller
    decide qué hacer con esa rama (devolver error 503, usar plano,
    etc.). El modelo raíz SÍ es obligatorio.
    """
    root_pair = load_one(settings.COLSIGN_ROOT_MODEL)

    by_root: dict = {}
    for group_name, model_name in settings.COLSIGN_SUB_MODELS.items():
        try:
            by_root[group_name] = load_one(model_name)
        except FileNotFoundError as exc:
            print(
                f"[WARN] Sub-modelo '{group_name}' ({model_name}) "
                f"no disponible: {exc}"
            )
            by_root[group_name] = None

    return {"root": root_pair, "by_root": by_root}


def load_flat_model() -> Optional[ModelPair]:
    """Carga el modelo plano de 154 clases. Devuelve `None` si falta."""
    try:
        return load_one(settings.COLSIGN_FLAT_MODEL)
    except FileNotFoundError as exc:
        print(
            f"[WARN] Modelo plano '{settings.COLSIGN_FLAT_MODEL}' "
            f"no disponible: {exc}"
        )
        return None


__all__ = [
    "ModelPair",
    "load_one",
    "load_hierarchical_set",
    "load_flat_model",
]
