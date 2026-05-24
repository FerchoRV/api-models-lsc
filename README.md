# Colsign API Models🤟🤖

Colsign API es el backend de una plataforma de investigación aplicada orientada a la inclusión social de la comunidad sorda. Utiliza inteligencia artificial para orquestar la traducción bidireccional entre texto, voz y la Lengua de Señas Colombiana (LSC).

Desarrollado con **FastAPI**, el sistema se conecta con modelos de deep learning locales para el procesamiento de señas y con la API de Google Gemini para la estructuración del sentido narrativo y la traducción conceptual.

## Características Principales

*   **Arquitectura Modular:** Separación estricta de responsabilidades mediante controladores (Endpoints), esquemas de validación (Pydantic) y capas de servicios.
*   **Gestión de Ciclo de Vida (Lifespan):** Carga eficiente en memoria de catálogos y recursos pesados al arrancar el servidor para optimizar tiempos de respuesta.
*   **Structured Outputs con Gemini:** Traducción estricta de texto a conceptos de señas restringidos a un catálogo de 566 señas permitidas, forzando respuestas en formato JSON mediante esquemas Pydantic.
*   **Sentido Narrativo Integrado:** Conversión de secuencias de señas crudas a oraciones fluidas, naturales y gramaticalmente correctas en español utilizando `gemini-2.5-flash`.

---

## Tecnologías Utilizadas

*   **Core:** Python 3.11+ / FastAPI
*   **Validación de Datos:** Pydantic v2 / Pydantic Settings
*   **Modelos de IA:** Google GenAI SDK (`google-genai`), Keras/TensorFlow (Modelos `.h5`)
*   **Servidor:** Uvicorn

---

## Estructura del Proyecto

```text
colsign-api/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   └── translation.py       # Endpoints de traducción (POST /text-to-sign)
│   │       └── router.py                # Enrutador central de la API (v1)
│   ├── core/
│   │   ├── config.py                    # Variables de entorno y configuraciones globales
│   │   └── lifespan.py                  # Eventos de inicio/apagado (Carga de JSON y modelos)
│   ├── information_json/
│   │   └── name_sign_list.json          # Catálogo oficial de las 566 señas disponibles
│   ├── schemas/
│   │   └── translation.py               # Modelos Pydantic para validar entradas y salidas
│   ├── services/
│   │   └── gemini_service.py            # Orquestación y lógica con Google GenAI
│   └── main.py                          # Punto de entrada de la aplicación FastAPI
├── models/                              # Carpeta para almacenar los modelos pesados de Keras
│   ├── actionAbecedario.h5
│   └── actionPalabrasV2.h5
├── .env                                 # Variables sensibles locales (Ignorado en git)
└── requirements.txt                     # Dependencias del proyecto
```
## Configuración e Instalación

1. Crear entorno virtual

python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

2. Instalar dependencias

pip install -r requirements.txt

3. Configurar variables de entorno

GEMINI_API_KEY="TU_API_KEY_DE_GOOGLE_AI_STUDIO"

4. Ejecución del Servidor

uvicorn app.main:app --reload

Documentación Interactiva
FastAPI autogenera la documentación de los endpoints. Puedes probar los flujos directamente en:

Swagger UI: http://127.0.0.1:8000/docs

## Endpoints Principales (v1)

POST /api/v1/text-to-sign
Toma una cadena de texto libre en español y devuelve una lista ordenada con los nombres de las señas del catálogo que logran interpretarla conceptualmente.

Cuerpo de la Petición (JSON):
```json
    {
      "text": "Hola, ¿cómo estás? Yo quiero comer"
    }
    ```

*   **Respuesta Exitosa (200 OK):**
```json

{
  "text_original": "Hola, ¿cómo estás? Yo quiero comer",
  "señas_seleccionadas": [
    "Hola",
    "Cómo estás",
    "Yo",
    "Querer",
    "c",
    "o",
    "m",
    "e",
    "r"
  ]
}
```

## 🤝 Contribuciones y Desarrollo
Este proyecto forma parte de una iniciativa de investigación aplicada para la inclusión social. Las sugerencias de optimización en el pipeline de procesamiento de los modelos y en las llamadas de contexto estructurado son bienvenidas.