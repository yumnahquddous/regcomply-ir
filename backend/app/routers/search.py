from fastapi import APIRouter, Query, Request, HTTPException
from app.schemas import SearchResponse

router = APIRouter()

@router.get("/search", response_model=SearchResponse)
def perform_search(request: Request, 
                   query: str = Query(..., description="The user search query"), 
                   system: str = Query("hybrid", description="Options: bm25, dense, hybrid")):
    try:
        engine = request.app.state.engine
        results = engine.search(query=query, system=system)
        return {"system": system.capitalize(), "results": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")