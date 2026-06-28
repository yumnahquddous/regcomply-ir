from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.engine import RetrievalEngine
from app.routers import search, document, generate

# Lifespan context manager loads ML models into app state on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- SERVER STARTUP SEQUENCE ---")
    engine = RetrievalEngine()
    engine.load_resources()
    app.state.engine = engine  # Attach engine to app state globally
    yield
    print("--- SERVER SHUTDOWN SEQUENCE ---")
    app.state.engine = None

app = FastAPI(title="RegComply-IR Backend", lifespan=lifespan)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321"],  # Astro frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Modular Routers
app.include_router(generate.router, prefix="/api", tags=["Answer Generation (RAG)"])
app.include_router(search.router, prefix="/api", tags=["Search"])
app.include_router(document.router, prefix="/api", tags=["Document Viewer"])

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RegComply-IR API is running!"}