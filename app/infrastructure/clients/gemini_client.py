import logging
from typing import Any
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
  
    async def hello(self) -> dict:
        return await self.generate("hello")
            
    async def generate(self, prompt:str, history:list|None = None) -> dict:
        try:
            response = await self.client.models.generate_content(
                model = self.model,
                contents = prompt
            )
            response_text = response.text or ""
            metadata:dict = {"model": self.model}
            if response.usage_metadata:
                metadata.update({
                    "prompt_token_count": response.usage_metadata.prompt_token_count,
                    "candidates_token_count": response.usage_metadata.candidates_token_count,
                    "total_token_count": response.usage_metadata.total_token_count})
            return {
                "status": "ok", 
                "answer": response_text, 
                "metadata": metadata}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
