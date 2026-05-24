from contextlib import asynccontextmanager
from fastapi import FastAPI
import tensorflow as tf
from app.core.config import settings
import json
import os
# Diccionario global donde se guardarán los modelos ya cargados en memoria
ai_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Cargando datos de memoria del sistema...")

    # 1. Cargamos el archivo JSON de manera segura usando un bloque 'with'
    try:
        if os.path.exists(settings.NAME_SIGN_PATH):
            with open(settings.NAME_SIGN_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Guardamos la lista directamente en el estado de la app de FastAPI
                app.state.name_sign_list = data.get("name", [])
            print(f"[INFO] Catálogo cargado con éxito. {len(app.state.name_sign_list)} señas disponibles.")
        else:
            print(f"[ERROR] No se encontró el archivo JSON en: {settings.NAME_SIGN_PATH}")
            app.state.name_sign_list = []
    except Exception as e:
        print(f"[ERROR] Error al leer el archivo JSON: {e}")
        app.state.name_sign_list = []

    try:
        
        # Aquí se abren los archivos usando las rutas de config.py
        ai_models["abecedario"] = tf.keras.models.load_model(settings.MODEL_ABECEDARIO_PATH)
        print(f"[OK] Modelo Abecedario cargado con éxito.")
        
        ai_models["palabras_v2"] = tf.keras.models.load_model(settings.MODEL_PALABRASV2_PATH)
        print(f"[OK] Modelo PalabrasV2 cargado con éxito.")
        
        ai_models["status"] = "Todos los modelos cargados exitosamente"
        print(f"[OK] Nombre de signos cargado con éxito.")
    except Exception as e:
        print(f"[ERROR CRÍTICO] Falló la carga de los modelos .h5: {e}")
        ai_models["status"] = f"Error en la carga: {str(e)}"
        
    yield  # Aquí la API se queda encendida esperando peticiones
    
    print("[INFO] Liberando recursos de memoria...")
    ai_models.clear()
    app.state.name_sign_list.clear()
    tf.keras.backend.clear_session()
    print("[INFO] Memoria RAM/GPU liberada correctamente.")