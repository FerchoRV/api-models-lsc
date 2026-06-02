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

    # ---- Modelos ColSign LSTM (jerárquico v2 + plano) ----
    # Directorios donde viven los checkpoints `.keras` y los `*_labels.json`.
    # `utils_pipeplanes.load_model()` resuelve automáticamente el `_best.keras`
    # a partir del nombre canónico (sin sufijo).
    MODELS_DIR: str = os.path.join(BASE_DIR, "app", "models")
    INFO_MODELS_DIR: str = os.path.join(BASE_DIR, "app", "info_models")

    # Nombre canónico del modelo raíz: clasifica el grupo morfológico (4 clases).
    COLSIGN_ROOT_MODEL: str = "colsign_lstm_norm_raiz_45_154_v2"

    # Sub-modelos por grupo predicho por el modelo raíz. Las claves DEBEN
    # coincidir exactamente con las etiquetas que produce el raíz.
    COLSIGN_SUB_MODELS: dict[str, str] = {
        "Grupo Estático":                     "colsign_lstm_norm_estatic_45_154_v2",
        "Grupo Dinámico Unimanual":           "colsign_lstm_norm_unimanual_45_154_v2",
        "Grupo Dinámico Bimanual Simétrico":  "colsign_lstm_norm_bi_simetrico_45_154",
        "Grupo Dinámico Bimanual Asimétrico": "colsign_lstm_norm_bi_asimetrico_45_154",
    }

    # Modelo plano: 154 clases en una sola red (fallback / comparación).
    COLSIGN_FLAT_MODEL: str = "colsign_lstm_norm_45_154"

    # ---- Concurrencia ----
    # Tamaño del threadpool que FastAPI/Starlette usa para correr endpoints
    # síncronos (`def`) y funciones envueltas con `run_in_threadpool`.
    # El default de AnyIO es `min(32, cpu_count + 4)`, lo que en Cloud Run
    # con 2 CPU son apenas ~6 threads. Subirlo a 64 permite atender más
    # requests concurrentes a los endpoints individuales (testing_models)
    # cuando el sistema recibe ráfagas de usuarios.
    # NOTA Cloud Run: no aumenta el coste; solo eleva el techo de
    # concurrencia. Si todos los threads están ocupados al mismo tiempo
    # el límite real pasa a ser la CPU del contenedor.
    THREADPOOL_SIZE: int = 64

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore" # Ignora otras variables que tengas en el .env que no use esta clase
    )

# Instanciamos para poder importarlo fácilmente
settings = Settings()