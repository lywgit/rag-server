import logging 
from fastapi import Request
from app.services.rag_service import RagService
from app.core.config import MAX_USER_INPUT

logger = logging.getLogger(__name__)

# utility functions
def truncate_user_input(query:str, max_len:int = MAX_USER_INPUT) -> str:
    if len(query) > max_len:
        logger.warning(f"User query truncated: {len(query)} -> {max_len}")
        query = query[:max_len]
    return query

def get_llm_client(request:Request) -> RagService:
    return request.app.state.llm_client

def get_rag_service(request:Request) -> RagService:
    return request.app.state.rag_service