# app/core/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Colsign API"
    VERSION: str = "1.0.0"
    
    # Tus Origins exactos de Flask migrados a FastAPI
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "https://www.colsign.com.co",
        "https://colsigns-app.vercel.app"
    ]
    
    # Rutas de tus modelos .h5 (compatibles con Cloud Run)
    # Usamos os.path para evitar problemas de rutas relativas según dónde se levante el contenedor
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    MODEL_ABECEDARIO_PATH: str = os.path.join(BASE_DIR, "app", "models_lsc", "actionAbecedario.h5")
    MODEL_PALABRASV2_PATH: str = os.path.join(BASE_DIR, "app", "models_lsc", "actionPalabrasV2.h5")
    NAME_SIGN_PATH: str = os.path.join(BASE_DIR, "app", "information_json", "name_sign_list.json")
    GEMINI_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore" # Ignora otras variables que tengas en el .env que no use esta clase
    )

# Instanciamos para poder importarlo fácilmente
settings = Settings()