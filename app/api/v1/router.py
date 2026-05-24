# app/api/v1/router.py
from fastapi import APIRouter

# Importamos el router del endpoint que acabamos de escribir
from app.api.v1.endpoints.translation import router as translation_router

api_router = APIRouter()

# Incluimos las rutas de traducción y les ponemos una etiqueta decorativa para Swagger
api_router.include_router(translation_router, tags=["Traducción"])

# Si mañana creas api/v1/endpoints/testing.py, lo importarías y lo incluirías aquí abajo:
# from app.api.v1.endpoints.testing import router as testing_router
# api_router.include_router(testing_router, prefix="/test", tags=["Pruebas Individuales"])