"""
index.json format:
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

import json
from domain.models import Document
from core.config import INDEX_JSON_PATH


def load_index_json_file(path:str = INDEX_JSON_PATH) -> list[Document]:
    with open(path, "r") as f:
        items = json.load(f)
    print(f"File loaded: {path}")
    print(f"Found {len(items)} items")
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
    print(f"Form {len(documents)} documents")
    return documents

    
if __name__ == "__main__":
    load_index_json_file()