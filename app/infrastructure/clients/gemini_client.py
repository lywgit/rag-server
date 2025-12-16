import logging
from google import genai
from app.infrastructure.clients.llm_client_interface import AsyncLLMClientInterface

logger = logging.getLogger(__name__)

class AsyncGeminiClient(AsyncLLMClientInterface):
    def __init__(self, model:str, api_key:str|None):
        if api_key is None:
            raise OSError("Environment variable 'GEMINI_API_KEY' is not set.")
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key).aio # aio makes the client async
        self.model = model
  
    async def hello(self) -> dict[str,str]:
        try:
            response = await self.client.models.generate_content(
                model=self.model,
                contents="hello"
            )
            response_text = response.text or ""
            return {"status": "ok", "answer": response_text}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
            
    async def generate(self, prompt:str, history:list|None = None) -> dict[str,str]:
        response = await self.client.models.generate_content(
            model = self.model,
            contents = prompt
        )
        return {"status": "ok", "answer": response.text or ""}
