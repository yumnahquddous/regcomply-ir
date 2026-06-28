from fastapi import APIRouter, Query, Request, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.llm import AnswerGenerator

router = APIRouter()
generator = AnswerGenerator()

class RAGResponse(BaseModel):
    query: str
    generated_answer: str
    sources_used: List[str] # Returns the passage IDs used to generate the answer

@router.get("/generate", response_model=RAGResponse)
def generate_rag_answer(
    request: Request, 
    query: str = Query(..., description="The user compliance query"),
    k: int = Query(5, description="Number of passages to retrieve for context")
):
    try:
        # 1. Subtask 1: Retrieve Relevant Passages (Using your champion Hybrid model)
        engine = request.app.state.engine
        retrieved_results = engine.search(query=query, system="hybrid", k=k)
        
        if not retrieved_results:
            return RAGResponse(
                query=query,
                generated_answer="No relevant regulatory documents were found for this query.",
                sources_used=[]
            )

        # 2. Subtask 2: Generate Answer
        answer = generator.generate_answer(query=query, retrieved_passages=retrieved_results)
        
        # Extract sources for citation transparency
        sources = [res['passage_id'] for res in retrieved_results]

        return RAGResponse(
            query=query,
            generated_answer=answer,
            sources_used=sources
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")