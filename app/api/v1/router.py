# app/api/v1/router.py
from fastapi import APIRouter

from app.api.v1.endpoints.testing_models import router as testing_models_router
from app.api.v1.endpoints.translation import router as translation_router

api_router = APIRouter()

# Rutas de traducción texto↔señas (texto→señas Gemini + sign-to-text plano y jerárquico).
api_router.include_router(translation_router, tags=["Traducción"])

# Rutas de prueba individual por modelo (raíz + 4 sub-modelos).
api_router.include_router(
    testing_models_router,
    prefix="/test",
    tags=["Pruebas Individuales"],
)
