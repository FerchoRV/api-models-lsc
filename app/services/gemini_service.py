# app/services/gemini_service.py
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from app.core.config import settings
# 1. Definimos la estructura exacta que queremos que Gemini devuelva para Text-to-Sign.
# Usamos Pydantic para forzar a la IA a responder estrictamente en este formato JSON.
class SignLanguageOutput(BaseModel):
    chosen_signs: list[str] = Field(
        description="Lista ordenada con los nombres de las señas que mejor transmiten el mensaje."
    )

class GeminiService:
    def __init__(self):
        # Inicializa el cliente. Buscará automáticamente la variable GEMINI_API_KEY en tu entorno/.env
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = "gemini-2.5-flash"  # Modelo ideal: ultra rápido y económico

    def get_narrative_sense(self, words_list: list[str]) -> str:
        """
        [Flujo Sign-to-Text]
        Toma la lista cruda de señas predichas por tus modelos de Keras
        y le da coherencia y sentido narrativo en español.
        """
        raw_words = ", ".join(words_list)
        
        # Configuración del "Gem" para dar sentido narrativo
        config = types.GenerateContentConfig(
            system_instruction=(
                "Actúas como un intérprete experto en Lengua de Señas. Tu tarea es recibir una lista "
                "de palabras clave capturadas secuencialmente desde un video y estructurarlas en una "
                "oración o párrafo en español que sea gramaticalmente correcto, natural y fluido. "
                "Conserva la intención original del mensaje. Devuelve ÚNICAMENTE la frase corregida, "
                "sin ningún tipo de introducción, saludo, explicación o notas adicionales."
            ),
            temperature=0.3, # Un poco de creatividad para que suene natural
        )
        
        prompt = f"Palabras clave detectadas: {raw_words}"
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        return response.text.strip()

    def select_signs_from_text(self, text: str, available_signs: list[str]) -> list[str]:
        """
        [Flujo Text-to-Sign]
        Recibe un texto común (máx 100 caracteres) y selecciona las señas correctas
        restringiéndose estrictamente al catálogo de tus 566 señas disponibles.
        """
        
        # Configuración del "Gem" para Text-to-Sign empleando Structured Outputs
        config = types.GenerateContentConfig(
            system_instruction=(
                "Actúas como un traductor automático de texto a señas. Tu trabajo es analizar la frase del "
                "usuario y descomponerla en los conceptos/señas que mejor transmitan el significado del mensaje. "
                "Debes elegir los nombres de las señas basándote exclusivamente en el catálogo permitido que "
                "te proporcionará el usuario. No inventes nombres de señas que no estén en la lista. "
                "El orden de la lista devuelta debe corresponder a la estructura lógica del mensaje en señas."
                "Si detectas una palabra que no se pueda traducir a señas, devuelve el deletro de esta, en caso de ser palabras grosera u ofensiva devuleve tres puntos suspensivos"
                "Si en el mensaje hay numeros debes devolver el numero separado por cada digito en caso de que sea un numero de telefono debes devolver el numero separado por cada digito y el signo de + al principio"
                "Devuelve ÚNICAMENTE la lista de señas, sin ningún tipo de introducción, saludo, explicación o notas adicionales."
                "Ejemplo de respuesta: ['Hola','Como estas','d','i','e','g','o']"
                f"El catálogo de señas permitidas es(Elige solo de aquí): {', '.join(available_signs)}"
            ),
            temperature=0.1,  # Temperatura muy baja para máxima precisión y cero inventiva
            response_mime_type="application/json",
            response_schema=SignLanguageOutput, # Forzamos la estructura Pydantic
        )
        
        # Le pasamos la frase del usuario y el catálogo disponible en el prompt
        prompt = (
            f"Frase del usuario a traducir: '{text}'\n\n"
            #f"Catálogo de señas permitidas (Elige solo de aquí):\n{available_signs}"
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        # Como forzamos el formato JSON, response.text será un string JSON válido como:
        # '{"chosen_signs": ["HOLA", "COMO", "ESTAS"]}'
        try:
            result_data = json.loads(response.text)
            return result_data.get("chosen_signs", [])
        except Exception as e:
            print(f"[ERROR] No se pudo parsear el JSON de Gemini: {e}")
            return []