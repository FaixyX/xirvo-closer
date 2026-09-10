from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.db.session import async_engine

ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT_DIR / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await async_engine.dispose()


app = FastAPI(
    title="Xirvo Sales Closer",
    description="Phase 2 consultative sales closer and lead qualification agent, grounded in the Xirvo RAG knowledge base.",
    lifespan=lifespan,
)
app.include_router(health_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
