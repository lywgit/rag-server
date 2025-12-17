import os
from pathlib import Path
from dotenv import load_dotenv
import logging 

# Configure app Logger 
logger = logging.getLogger('app')

logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt='[%(levelname)s] %(name)s: %(message)s'
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# module logger 
logger = logging.getLogger(__name__)
load_dotenv()

# Required Secrets
GEMINI_API_KEY =  os.environ.get("GEMINI_API_KEY","").strip()

# Required Variable 
INDEX_JSON_URL = os.environ.get("INDEX_JSON_URL","").strip()
if not INDEX_JSON_URL.startswith("http"):
    INDEX_JSON_URL =  str(Path(INDEX_JSON_URL).resolve())

# Optional settings  (can be overridden by env variable)\
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_SENTENCE_ENCODER_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MAX_USER_INPUT = 1000
DEFAULT_FILE_CACHE_DIR = "./cache"
DEFAULT_SEARCH_LIMIT = 10

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
SENTENCE_ENCODER_MODEL = os.environ.get("SENTENCE_ENCODER_MODEL", DEFAULT_SENTENCE_ENCODER_MODEL)
MAX_USER_INPUT = int(os.environ.get("MAX_USER_INPUT", DEFAULT_MAX_USER_INPUT))
FILE_CACHE_DIR = str(Path(os.environ.get("FILE_CACHE_DIR", DEFAULT_FILE_CACHE_DIR)).resolve())
SEARCH_LIMIT = int(os.environ.get("SEARCH_LIMIT", DEFAULT_SEARCH_LIMIT))


# Settings (no associated env variable)
BM25_K1 = 1.5
BM25_B = 0.75 
HYBRID_RRF_K = 60.0