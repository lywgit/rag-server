import os
from dotenv import load_dotenv
load_dotenv()

INDEX_JSON_PATH = "../data/index.json"

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_SENTENCE_ENCODER_MODEL = "all-MiniLM-L6-v2"

GEMINI_API_KEY =  os.environ.get("GEMINI_API_KEY")

FILE_CACHE_DIR = "./cache"