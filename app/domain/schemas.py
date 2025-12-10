# Pydantic: QueryRequest, QueryResponse, Source

from pydantic import (
    BaseModel,
    Field
)

class HealthRequest:
    pass   

class QueryRequest:
    prompt:str = Field(..., description="The user's query prompt")
    top_k:int = Field(3, description="Number of top relevant sources to retrieve")  
    