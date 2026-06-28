from pydantic import BaseModel
from typing import List

class SearchResultItem(BaseModel):
    passage_id: str
    document_id: int
    document_name: str
    text: str
    score: float

class SearchResponse(BaseModel):
    system: str
    results: List[SearchResultItem]

class DocumentResponse(BaseModel):
    filename: str
    content: str