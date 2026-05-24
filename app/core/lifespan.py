from contextlib import asynccontextmanager
from fastapi import FastAPI
import tensorflow as tf
from app.core.config import settings

# Diccionario global donde se guardarán los modelos ya cargados en memoria
ai_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Cargando modelos de Keras en memoria RAM para inferencia...")
    try:
        # Aquí se abren los archivos usando las rutas de config.py
        ai_models["abecedario"] = tf.keras.models.load_model(settings.MODEL_ABECEDARIO_PATH)
        print(f"[OK] Modelo Abecedario cargado con éxito.")
        
        ai_models["palabras_v2"] = tf.keras.models.load_model(settings.MODEL_PALABRASV2_PATH)
        print(f"[OK] Modelo PalabrasV2 cargado con éxito.")
        
        ai_models["status"] = "Todos los modelos cargados exitosamente"
    except Exception as e:
        print(f"[ERROR CRÍTICO] Falló la carga de los modelos .h5: {e}")
        ai_models["status"] = f"Error en la carga: {str(e)}"
        
    yield  # Aquí la API se queda encendida esperando peticiones
    
    print("[INFO] Liberando recursos de memoria...")
    ai_models.clear()
    tf.keras.backend.clear_session()
    print("[INFO] Memoria RAM/GPU liberada correctamente.")