# app/schemas/translation.py
from pydantic import BaseModel, Field

# Lo que el usuario nos envía en el cuerpo (Body) del POST
class TextToSignRequest(BaseModel):
    text: str = Field(..., max_length=100, description="Texto a traducir a señas")

# Lo que nuestra API le responde al usuario
class TextToSignResponse(BaseModel):
    text_original: str
    señas_seleccionadas: list[str]