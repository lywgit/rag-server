from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class Document:
    id:str
    content:str
    metadata:dict # metadata of individual document

@dataclass
class SearchResult:
    document:Document
    score:float
    rank:int
    
  
    

class QueryRequest(BaseModel):
    pass

class QueryResponse(BaseModel):
    pass
    