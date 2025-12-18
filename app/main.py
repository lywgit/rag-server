import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from app.api.query import query_router
from app.api.health import health_router
from app.services.rag_service import RagService
from app.infrastructure.retriever import FileRetriever
from app.infrastructure.repositories.file_vector_store import FileVectorStore
from app.infrastructure.repositories.bm25_keyword_store import BM25KeywordStore
from app.infrastructure.repositories.tokenizers import JiebaTokenizer
from app.infrastructure.clients.gemini_client import AsyncGeminiClient
from app.infrastructure.embeddings.embedder import SentenceTransformerEmbedder
from app.core.config import CORS_DOMAIN_NAME

from app.core.config import (
    GEMINI_MODEL,
    GEMINI_API_KEY,
    SENTENCE_ENCODER_MODEL,
    INDEX_JSON_URL
    )
from app.infrastructure.ingestion.parser import load_documents

# Initialize service 
logger = logging.getLogger(__name__)
logger.info("Initializing service")

retriever = FileRetriever(
    SentenceTransformerEmbedder(SENTENCE_ENCODER_MODEL),
    FileVectorStore(),
    BM25KeywordStore(JiebaTokenizer())
    )
llm_client = AsyncGeminiClient(GEMINI_MODEL, GEMINI_API_KEY)
rag_service = RagService(retriever, llm_client)

logger.info(f"Loading index.json from {INDEX_JSON_URL}")

def build_rag_service():
    docs = load_documents(INDEX_JSON_URL)
    rag_service.build(docs)
    logger.info(f"RAG Service built {datetime.datetime.now()}, total {len(docs)} documents")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Service")
    try:
        build_rag_service()
    except Exception as e:
        logger.exception(f"Failed to build RAG Service: {e}", exc_info=True)
        raise
    scheduler = BackgroundScheduler()
    scheduler.add_job(build_rag_service, "interval", days = 1)
    scheduler.start()
    yield
       
    logger.info("RAG Service shutdown")

app = FastAPI(lifespan=lifespan)
app.state.rag_service = rag_service
app.state.llm_client = llm_client

cors_domain_name_list = CORS_DOMAIN_NAME.split(',')
logger.info(f'allow_origins: {cors_domain_name_list}')
app.add_middleware(
    CORSMiddleware,
    allow_origins= cors_domain_name_list,  # "*" for all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(health_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)

    
