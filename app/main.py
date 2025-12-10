from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.query import query_router
from api.healthz import healthz_router
from services.rag_service import RagService
from infrastructure.retriever import FileRetriever
from infrastructure.repositories.file_vector_store import FileVectorStore
from infrastructure.repositories.keyword_store_interface import BM25Store
from infrastructure.clients.gemini_client import GeminiClient
from infrastructure.embeddings.embedder import SentenceTransformerEmbedder
from core.config import DEFAULT_GEMINI_MODEL, DEFAULT_SENTENCE_ENCODER_MODEL

retriever = FileRetriever(
    SentenceTransformerEmbedder(DEFAULT_SENTENCE_ENCODER_MODEL),
    FileVectorStore(),
    BM25Store()
    )
llm_client = GeminiClient()
rag_service = RagService(retriever, llm_client)

from infrastructure.ingestion.parser import load_index_json_file

# Lifespan 事件處理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動邏輯
    try:
        docs = load_index_json_file()
        rag_service.build(docs)
        print("RAG Service started")
    except Exception as e:
        print(f"Failed to build RAG Service: {e}")
        raise
    
    yield
    
    # 關閉邏輯
    print("✓ RAG Service shutdown")

app = FastAPI(lifespan=lifespan)
app.state.rag_service = rag_service
app.include_router(healthz_router)
app.include_router(query_router)


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q:str|None = None):
#     return {"item_id": item_id, "q": q}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)

    
