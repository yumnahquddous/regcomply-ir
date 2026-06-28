import os
from fastapi import APIRouter, Query, HTTPException
from app.schemas import DocumentResponse
from app.config import TXT_DOCS_DIR, DOCUMENT_MAPPING

router = APIRouter()

@router.get("/document", response_model=DocumentResponse)
def get_full_document(filename: str = Query(..., description="The ID or filename to load")):
    safe_filename = os.path.basename(filename)
    
    # Try resolving via the numeric mapping dictionary first
    try:
        # Check if the incoming argument is a numeric ID string like "23"
        if safe_filename.isdigit():
            doc_id = int(safe_filename)
            if doc_id in DOCUMENT_MAPPING:
                safe_filename = DOCUMENT_MAPPING[doc_id]
    except Exception:
        pass # Fallback to using raw filename directly if parsing fails

    # Append standard file extension if missing from filename string
    if not safe_filename.endswith(".txt"):
        safe_filename += ".txt"
        
    file_path = os.path.join(TXT_DOCS_DIR, safe_filename)
    print(f"🔍 DEBUG: Attempting to open document path -> {file_path}")
    
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Document '{safe_filename}' not found on file system paths."
        )
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
            
        return {"filename": safe_filename, "content": full_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file resource: {str(e)}")