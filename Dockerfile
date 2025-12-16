FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "uvicorn[standard]>=0.38.0"
RUN pip install --no-cache-dir -e .

# pre-download sentence transformer model and nltk stopwords
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "import nltk;  nltk.download('stopwords')"
COPY app/ ./app/
COPY data/ ./data/

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
