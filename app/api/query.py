from fastapi import APIRouter, Depends, Request
from services.rag_service import RagService
from infrastructure.clients.gemini_client import GeminiClient

def get_rag_service(request:Request) -> RagService:
    return request.app.state.rag_service

query_router = APIRouter(prefix="/query", tags=["rag_query"])

# check rag service health
@query_router.get('/rag-health')
def gemini_api_health():
    gemini_client = GeminiClient()
    return gemini_client.ping()

@query_router.get('/quick_query/{query}')
def quick_query(query:str, rag_service = Depends(get_rag_service)):
    return rag_service.answer(query)