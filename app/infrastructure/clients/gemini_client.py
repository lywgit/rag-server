from google import genai
from google.genai import types
from core.config import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_API_KEY
)
from .llm_client_interface import LLMClientInterface

class GeminiClient(LLMClientInterface):
    def __init__(self, model:str = DEFAULT_GEMINI_MODEL, api_key:str|None = GEMINI_API_KEY):
        if api_key is None:
            raise OSError("Environment variable 'GEMINI_API_KEY' is not set.")
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.model = model
  
    def ping(self) -> dict[str,str]:
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents="ping"
            )
            response_text = response.text or ""
            return {"status": "ok", "detail": response_text}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
            

    def generate(self, prompt:str, history:list|None = None) -> str:
        response = self.client.models.generate_content(
            model = self.model,
            contents = prompt
        )
        return response.text or ""
    

# def get_gemini_client_async() -> AsyncClient:
#     load_dotenv()
#     api_key = os.environ.get("GEMINI_API_KEY")
#     if api_key is None:
#         raise OSError("Environment variable 'GEMINI_API_KEY' is not set.")
#     client = genai.Client(api_key=api_key).aio
#     return client