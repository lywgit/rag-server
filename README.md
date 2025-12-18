# RAG Server

A retrieval-augmented generation (RAG) API built with FastAPI, designed to serve semantic and keyword-based search over document indexes with LLM-powered answer generation. This server ingests a JSON index of documents at startup, builds vector and keyword indexes, and exposes endpoints for search and question-answering workflows.

Configured for deployment on Google Cloud Run with API Gateway support, it includes quota management, structured logging, and environment-based configuration.


## Features

- FastAPI endpoints: readiness/health, LLM hello, direct LLM query, document search, and RAG (search + answer).
- Layered architecture: `api` (routes), `services` (orchestration), `domain` (models/schemas), `infrastructure` (clients/embeddings/repositories/ingestion), and `core` (config/logging).
- Retrieval options: semantic search (SentenceTransformer `all-MiniLM-L6-v2`), keyword search (BM25), and a hybrid mode for combining results.
- LLM integration: Google Gemini via `google-genai`; async client available with basic usage metadata in responses.
- Startup build process: ingestion parses `index.json` (local path or HTTP) and builds keyword/vector indexes during app lifespan startup.
- Configuration by environment: model choices and paths (`GEMINI_API_KEY`, `GEMINI_MODEL`, `SENTENCE_ENCODER_MODEL`, `INDEX_JSON_URL`, `PORT`); path resolution for local files; CORS configured.
- Containerized deployment: Dockerfile optimized for Cloud Run; app respects `PORT` and logs to stdout.
- API gateway mapping: OpenAPI (`openapi-run.yaml`) with `x-google-backend` routing and per-endpoint quotas.
- Logging: application logger writes structured messages to stdout for aggregation by Cloud Run/Cloud Logging.

## Directory Structure

```
.
├── app/
│   ├── main.py                     # FastAPI app bootstrap + lifespan/DI
│   ├── api/                        # HTTP routes
│   │   ├── health.py               # Readiness/health endpoint
│   │   ├── query.py                # LLM hello, direct LLM, search, RAG
│   │   └── utils.py                # API helpers/utilities
│   ├── core/
│   │   └── config.py               # Env loading, path resolution, logging
│   ├── domain/
│   │   ├── models.py               # Domain entities/value objects
│   │   └── schemas.py              # Pydantic request/response DTOs
│   ├── infrastructure/
│   │   ├── clients/
│   │   │   ├── gemini_client.py    # Async Gemini client via google-genai
│   │   │   └── llm_client_interface.py
│   │   ├── embeddings/
│   │   │   └── embedder.py         # SentenceTransformer wrapper
│   │   ├── ingestion/
│   │   │   └── parser.py           # Load index.json → Documents
│   │   ├── repositories/
│   │   │   ├── file_vector_store.py
│   │   │   ├── bm25_keyword_store.py
│   │   │   ├── vector_store_interface.py
│   │   │   ├── keyword_store_interface.py
│   │   │   ├── tokenizer_interface.py
│   │   │   ├── tokenizers.py
│   │   │   └── tokenizer_data/
│   │   │       ├── dict.txt.big
│   │   │       └── stopwords.txt
│   │   └── retriever.py            # Keyword/semantic/hybrid retrieval orchestration
│   └── services/
│       └── rag_service.py          # Orchestrates retriever + LLM for answers
├── data/
│   └── index.json                  # Sample index input
├── Dockerfile
├── openapi-run.yaml                # API Gateway config (x-google-*)
├── pyproject.toml
├── docker-run.sh
├── test_quota.sh
├── ROADMAP.md
└── README.md
```

- app/api: request handling and HTTP route composition only.
- app/services: business orchestration (RAG flow), no I/O details.
- app/domain: core types and Pydantic schemas used by APIs/services.
- app/infrastructure: integrations (LLM client, embeddings), storage (vector/BM25), retrieval logic, ingestion.
- app/core: configuration, environment resolution, and logging setup.

## Quick Start

- Prerequisites: Python 3.12, a Gemini API key, and an `index.json` source (file path or URL).
- Environment variables:
	- `GEMINI_API_KEY`: required
	- `INDEX_JSON_URL`: http(s) URL or local path to index.json
	- `GEMINI_MODEL` (default: `gemini-2.5-flash`)
	- `SENTENCE_ENCODER_MODEL` (default: `all-MiniLM-L6-v2`)
	- `PORT` (default: 8000)

### Run Locally

Using pip:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --no-cache-dir -e .
pip install --no-cache-dir "uvicorn[standard]>=0.38.0"

export GEMINI_API_KEY="<your-key>"
export INDEX_JSON_URL="./data/index.json"  # or an https URL
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Using uv:

```bash
uv sync --no-dev
export GEMINI_API_KEY="<your-key>"
export INDEX_JSON_URL="./data/index.json"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

Build and run:

```bash
docker build -t rag-server:latest .
docker run --env-file .env -p 8000:8000 rag-server:latest
# or pass envs explicitly
docker run -e GEMINI_API_KEY="<your-key>" -e INDEX_JSON_URL="/app/data/index.json" -p 8000:8000 rag-server:latest
```

Container respects `PORT` (default 8000). Cloud Run will override `PORT` automatically.

### Cloud Run

- Push the image to Artifact Registry and deploy Cloud Run with `PORT` and secrets set.
- Logs are emitted to stdout/stderr and visible in Cloud Logging.
- Optionally front with API Gateway using `openapi-run.yaml` for per-endpoint quotas.

## API Examples

Health/readiness:

```bash
curl -s http://localhost:8000/
```

LLM hello:

```bash
curl -s http://localhost:8000/query/llm/hello
```

Direct LLM query:

```bash
curl -s -X POST "http://localhost:8000/query/llm?query=Hello"
```

Search:

```bash
curl -s -X POST http://localhost:8000/query/search \
	-H "Content-Type: application/json" \
	-d '{"query":"your question","top_k":5,"method":"semantic"}'
```

RAG (search + answer):

```bash
curl -s -X POST http://localhost:8000/query/rag \
	-H "Content-Type: application/json" \
	-d '{"query":"your question","top_k":5,"method":"hybrid"}'
```

## Notes

- On startup, the app loads `INDEX_JSON_URL` and builds indexes; failures are logged and surface during readiness.
- For AMD64 builds, PyTorch CPU wheels are larger than ARM; Docker image size varies accordingly.


## Author's Note

- This project was built with extensive help from AI's advices, but it was not written by an AI agent.
- Most of this document was AI-assisted.
- The core RAG logic draws on my completed implementation of the Boot.dev course [Learn Retrieval Augmented Generation](https://www.boot.dev/courses/learn-retrieval-augmented-generation), see: [lywgit/bootdev-rag](https://github.com/lywgit/bootdev-rag). Notable changes in this repository include:
	1. A complete server-side redesign using a layered architecture.
	2. Chinese tokenization via Jieba.
	3. No chunking; documents are indexed as-is.
- This app does not use a managed vector database, so it won’t handle very large corpora out of the box. Adopting a vector DB is hopefully straightforward given the repository interface design.

