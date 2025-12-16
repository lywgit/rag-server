import json
import logging
from http import HTTPStatus
from fastapi import APIRouter, Depends, Request, HTTPException
from app.infrastructure.clients.llm_client_interface import AsyncLLMClientInterface
from app.services.rag_service import RagService
from app.domain.schemas import SearchRequest
from app.core.config import MAX_USER_INPUT, SEARCH_LIMIT

logger = logging.getLogger(__name__)

def get_llm_client(request:Request) -> RagService:
    return request.app.state.llm_client

def get_rag_service(request:Request) -> RagService:
    return request.app.state.rag_service

query_router = APIRouter(prefix="/query", tags=["query"])

# utility functions
def truncate_user_input(query:str, max_len:int = MAX_USER_INPUT) -> str:
    if len(query) > max_len:
        logger.warning(f"User input text truncated: {len(query)} -> {max_len}")
        query = query[:max_len]
    return query


# Ping llm as health check of underlying API service
@query_router.get('/llm/hello', status_code=201, description="Test LLM client by sending a 'hello' message")
async def hello(llm_client:AsyncLLMClientInterface = Depends(get_llm_client)):
    logger.debug("Hello LLM")
    try:
        return await llm_client.hello()
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=str(e)
        )

@query_router.post('/llm', status_code=201, description="Send query to LLM client")
async def llm_query(query:str, llm_client:AsyncLLMClientInterface = Depends(get_llm_client)):
    logger.debug("Send query to LLM client")
    query = truncate_user_input(query)
    try: 
        return {
            "answer": await llm_client.generate(query)
            }
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=str(e)
            )

@query_router.post('/search', status_code=201)
async def search_query(request: SearchRequest, rag_service = Depends(get_rag_service)):
    logger.debug("Pure search query")
    query = truncate_user_input(request.query)
    try:
        response = await rag_service.search(query, top_k=request.top_k, method=request.method)
        logger.debug(f"Search returned {len(response['search_result'])} results")
        return {
            "search_result": response["search_result"]
        }
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=str(e)
            )


@query_router.post('/rag', status_code=201)
async def rag_query(request: SearchRequest, rag_service = Depends(get_rag_service)):
    logger.debug("RAG search query")
    query = truncate_user_input(request.query)
    try:
        response = await rag_service.answer(query, top_k=request.top_k, method=request.method)
        return {
            "search_result": response["search_result"],
            "answer": response["answer"]
        }
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=str(e)
            )
