import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

load_dotenv()

from app.rag_pipeline import RAGPipeline
from app.logger import logger

rag: RAGPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    logger.info("Initializing RAG pipeline...")
    rag = RAGPipeline()
    rag.load_or_build_vectorstore()
    logger.info("RAG pipeline ready.")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Production RAG Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ─────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

# ── Models ─────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float
    chunks_retrieved: int

class UploadResponse(BaseModel):
    message: str
    chunks_indexed: int

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "vectorstore_docs": rag.doc_count() if rag else 0}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not rag:
        raise HTTPException(503, "RAG pipeline not ready")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    start = time.perf_counter()
    result = rag.query(req.question)
    latency = round((time.perf_counter() - start) * 1000, 1)
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=latency,
        chunks_retrieved=result["chunks_retrieved"],
    )

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not rag:
        raise HTTPException(503, "RAG pipeline not ready")
    async def token_generator() -> AsyncGenerator[str, None]:
        async for token in rag.stream_query(req.question):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(token_generator(), media_type="text/event-stream")

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(400, "Only PDF, TXT, and MD files are supported")
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, file.filename)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    chunks_added = rag.index_file(save_path)
    return UploadResponse(
        message=f"Successfully indexed {file.filename}",
        chunks_indexed=chunks_added,
    )

@app.get("/eval/run")
async def run_evaluation():
    scores = rag.evaluate()
    return {"scores": scores, "status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)