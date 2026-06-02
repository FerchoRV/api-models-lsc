from contextlib import asynccontextmanager
from fastapi import FastAPI
import anyio.to_thread
import tensorflow as tf
from app.core.config import settings
from app.core import model_registry
import json
import os

# Diccionario global donde se guardan los modelos ya cargados en memoria.
# Estructura esperada tras un arranque exitoso:
#
#   ai_models["abecedario"]        -> keras.Model legacy
#   ai_models["palabras_v2"]       -> keras.Model legacy
#   ai_models["colsign_jerarquico"] -> {
#       "root":    (keras.Model, ModelInfo),
#       "by_root": {grupo: (keras.Model, ModelInfo) | None, ...},
#   }
#   ai_models["colsign_plano"]     -> (keras.Model, ModelInfo) | ausente
#   ai_models["status"]            -> str con el resultado global
ai_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Cargando datos de memoria del sistema...")

    # 0) Ajustar el threadpool de Starlette/AnyIO.
    # FastAPI ejecuta endpoints `def` (síncronos) y `run_in_threadpool` sobre
    # este pool. El default es bajo (~6-8 threads en Cloud Run con 2 CPUs),
    # lo que genera colas cuando hay ráfagas de usuarios en los endpoints
    # individuales de `testing_models`. Lo subimos para soportar 50+
    # requests concurrentes sin esperar turno en cola.
    try:
        anyio.to_thread.current_default_thread_limiter().total_tokens = (
            settings.THREADPOOL_SIZE
        )
        print(f"[INFO] Threadpool ajustado a {settings.THREADPOOL_SIZE} tokens.")
    except Exception as e:
        print(f"[WARN] No se pudo ajustar el threadpool: {e}")

    # 1) Catálogo de nombres de señas (consumido por el endpoint Gemini)
    try:
        if os.path.exists(settings.NAME_SIGN_PATH):
            with open(settings.NAME_SIGN_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                app.state.name_sign_list = data.get("name", [])
            print(
                f"[INFO] Catálogo cargado con éxito. "
                f"{len(app.state.name_sign_list)} señas disponibles."
            )
        else:
            print(f"[ERROR] No se encontró el archivo JSON en: {settings.NAME_SIGN_PATH}")
            app.state.name_sign_list = []
    except Exception as e:
        print(f"[ERROR] Error al leer el archivo JSON: {e}")
        app.state.name_sign_list = []

    load_errors: list[str] = []

    # 2) Modelos legacy (.h5) — mantenidos por compatibilidad con la
    # versión anterior de la API. Si fallan, el resto sigue cargando.
    try:
        ai_models["abecedario"] = tf.keras.models.load_model(settings.MODEL_ABECEDARIO_PATH)
        print("[OK] Modelo Abecedario cargado con éxito.")

        ai_models["palabras_v2"] = tf.keras.models.load_model(settings.MODEL_PALABRASV2_PATH)
        print("[OK] Modelo PalabrasV2 cargado con éxito.")
    except Exception as e:
        msg = f"Fallo carga de modelos legacy (.h5): {e}"
        print(f"[ERROR] {msg}")
        load_errors.append(msg)

    # 3) Set jerárquico v2: modelo raíz (4 grupos) + 4 sub-modelos.
    # Comparte input shape (45, 225), por lo que el endpoint puede
    # reutilizar las features extraídas entre raíz y sub-modelo.
    try:
        print("[INFO] Cargando set jerárquico v2 (raíz + 4 sub-modelos)...")
        ai_models["colsign_jerarquico"] = model_registry.load_hierarchical_set()

        _, root_info = ai_models["colsign_jerarquico"]["root"]
        print(
            f"[OK] Raíz: {root_info.name} "
            f"(K={root_info.num_classes}, input={root_info.input_shape})"
        )

        for group, pair in ai_models["colsign_jerarquico"]["by_root"].items():
            if pair is None:
                msg = f"Sub-modelo '{group}' no disponible."
                print(f"[WARN] {msg}")
                load_errors.append(msg)
                continue
            _, sub_info = pair
            print(
                f"[OK] Sub [{group}]: {sub_info.name} "
                f"(K={sub_info.num_classes})"
            )
    except Exception as e:
        msg = f"Fallo carga del set jerárquico v2: {e}"
        print(f"[ERROR] {msg}")
        load_errors.append(msg)

    # 4) Modelo plano (154 clases). Es opcional, sirve como fallback /
    # baseline; si no está, no bloqueamos el arranque.
    try:
        flat = model_registry.load_flat_model()
        if flat is not None:
            ai_models["colsign_plano"] = flat
            _, flat_info = flat
            print(
                f"[OK] Plano: {flat_info.name} "
                f"(K={flat_info.num_classes}, input={flat_info.input_shape})"
            )
    except Exception as e:
        msg = f"Fallo carga del modelo plano: {e}"
        print(f"[ERROR] {msg}")
        load_errors.append(msg)

    # Resumen global para el health check
    if load_errors:
        ai_models["status"] = (
            f"Cargado con {len(load_errors)} advertencia(s): "
            + " | ".join(load_errors)
        )
    else:
        ai_models["status"] = "Todos los modelos cargados exitosamente"

    yield  # Aquí la API se queda encendida esperando peticiones.

    print("[INFO] Liberando recursos de memoria...")
    ai_models.clear()
    if hasattr(app.state, "name_sign_list") and isinstance(app.state.name_sign_list, list):
        app.state.name_sign_list.clear()
    tf.keras.backend.clear_session()
    print("[INFO] Memoria RAM/GPU liberada correctamente.")
