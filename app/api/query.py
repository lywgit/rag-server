import logging
from http import HTTPStatus
from fastapi import APIRouter, Depends, HTTPException
from app.infrastructure.clients.llm_client_interface import AsyncLLMClientInterface
from app.domain.schemas import SearchRequest

from app.api.utils import get_llm_client, get_rag_service, truncate_user_input

logger = logging.getLogger(__name__)

query_router = APIRouter(prefix="/query", tags=["query"])

# Ping llm as health check of underlying API service
@query_router.get('/llm/hello', status_code=201, description="Test LLM client by sending a 'hello' message")
async def hello(llm_client:AsyncLLMClientInterface = Depends(get_llm_client)):
    logger.info("Hello LLM")
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
    logger.info(f"LLM query: {query[:50]}... length={len(query)}")
    query = truncate_user_input(query)
    try: 
        response = await llm_client.generate(query)
        return {
                "answer": response.get("answer",""),
                "metadata": response.get("metadata",{})
                }
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=str(e)
            )

@query_router.post('/search', status_code=201)
async def search_query(request: SearchRequest, rag_service = Depends(get_rag_service)):
    logger.info(f"Search query: {request.query[:50]}... length={len(request.query)}")
    query = truncate_user_input(request.query)
    try:
        response = await rag_service.search(query, top_k=request.top_k, method=request.method)
        logger.info(f"Search response: {len(response['search_result'])} results")
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
    logger.info(f"RAG query: {request.query[:50]}... length={len(request.query)}")
    query = truncate_user_input(request.query)
    try:
        response = await rag_service.answer(query, top_k=request.top_k, method=request.method)
        logger.info(f"RAG response meta={response.get('metadata','')}")
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
