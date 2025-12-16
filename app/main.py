from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.api.query import query_router
from app.api.health import healthz_router
from app.services.rag_service import RagService
from app.infrastructure.retriever import FileRetriever
from app.infrastructure.repositories.file_vector_store import FileVectorStore
from app.infrastructure.repositories.bm25_keyword_store import BM25KeywordStore
from app.infrastructure.repositories.tokenizers import JiebaTokenizer
from app.infrastructure.clients.gemini_client import AsyncGeminiClient
from app.infrastructure.embeddings.embedder import SentenceTransformerEmbedder

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        docs = load_documents(INDEX_JSON_URL)
        rag_service.build(docs)
        logger.info("RAG Service started")
    except Exception as e:
        logger.exception(f"Failed to build RAG Service: {e}", exc_info=True)
        raise
    yield
       
    logger.info("RAG Service shutdown")

app = FastAPI(lifespan=lifespan)
app.state.rag_service = rag_service
app.state.llm_client = llm_client

app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],  # Or specify your domain: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(healthz_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)

    
