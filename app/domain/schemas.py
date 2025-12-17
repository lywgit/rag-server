# Pydantic: QueryRequest, QueryResponse, Source

from pydantic import (
    BaseModel,
    Field
)
from app.core.config import SEARCH_LIMIT
from typing import Literal


class SearchRequest(BaseModel):
    query:str = Field(..., description="User's input")
    top_k:int = Field(default=SEARCH_LIMIT, description="Number of top relevant sources to retrieve") 
    method: Literal["semantic", "keyword", "hybrid"] = Field(default="hybrid", description="Search method: keyword (bm25), semantic, hybrid")

class RagResponse(BaseModel):
    answer: str
    