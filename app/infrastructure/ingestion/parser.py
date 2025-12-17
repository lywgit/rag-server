"""
index.json structure:
[
  {
    "content": "Welcome! I use this site to hold my profile and share some thoughts from time to time. Read more about me or check out my posts if you are interested.\n",
    "date": null,
    "permalink": "https://lywgit.github.io/",
    "section": "",
    "summary": "",
    "title": ""
  },
  ...
  ]
"""
import logging
import json
from app.domain.models import Document 
logger = logging.getLogger(__name__)


def fetch_index_json(path:str) -> list[dict]:
    """Fetch index.json from local file or url"""
    if path.startswith("http://") or path.startswith("https://"):
        logger.debug(f"Fetching index.json from remote url: {path}")
        import requests
        response = requests.get(path)
        response.raise_for_status()
        items = response.json()
    else:
        logger.debug(f"Fetching index.json from local file: {path}")
        with open(path, "r") as f:
            items = json.load(f)
    logger.debug(f"Fetched. Got {len(items)} items")
    return items

def format_documents(items) -> list[Document]:
    """Convert item into document format. Title and content are combined"""
    documents = []
    for item in items:
        if not item["content"]:
            continue
        documents.append(
            Document(
                id = item["permalink"],
                content = f"{item['title']} - {item['content']}",
                metadata = {k:v for k,v in item.items() if k !="content"}
                )
            )
    logger.debug(f"Format {len(items)} items and form {len(documents)} documents")
    return documents

def load_documents(path:str) -> list[Document]:
    """Load Document either from http(s) url or local json file path """
    items = fetch_index_json(path)
    return format_documents(items)