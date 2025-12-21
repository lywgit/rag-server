# from pydantic import BaseModel
# from dataclasses import dataclass

# @dataclass
# class Document:
#     id:str
#     content:str
#     metadata:dict # metadata of individual document

# @dataclass
# class SearchResult:
#     document:Document
#     score:float
#     rank:int

from pydantic import BaseModel

class Document(BaseModel):
    id: str
    content: str
    metadata: dict

class SearchResult(BaseModel):
    document: Document
    score: float
    rank: int