# app/api/v1/endpoints/translation.py
from fastapi import APIRouter, HTTPException, Request

from app.schemas.translation import TextToSignRequest, TextToSignResponse
from app.services.gemini_service import GeminiService

router = APIRouter()
gemini_service = GeminiService()

@router.post("/text-to-sign", response_model=TextToSignResponse)
def text_to_sign(payload: TextToSignRequest, request: Request):
    # 1. Accedemos directamente a la lista guardada en el estado de la app
    catalogo_nombres = getattr(request.app.state, "name_sign_list", [])
    
    # 2. Validamos que no esté vacía
    if not catalogo_nombres:
        raise HTTPException(
            status_code=500, 
            detail="El catálogo de señas no está disponible o no se pudo cargar en el servidor."
        )
    
    # 3. Llamamos al servicio de Gemini pasándole el texto y tu lista extraída del JSON
    señas_elegidas = gemini_service.select_signs_from_text(
        text=payload.text, 
        available_signs=catalogo_nombres  # Ya es una lista nativa de Python
    )
    
    # 4. Retornamos la respuesta
    return TextToSignResponse(
        text_original=payload.text,
        señas_seleccionadas=señas_elegidas
    )