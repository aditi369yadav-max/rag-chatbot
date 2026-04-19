"""
Production RAG Chatbot - Main FastAPI Application
"""

import os
import time
from dotenv import load_dotenv
load_dotenv()
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from app.rag_pipeline import RAGPipeline
from app.logger import logger


# Global RAG pipeline instance
rag: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline on startup."""
    global rag
    logger.info("Initializing RAG pipeline...")
    rag = RAGPipeline()
    rag.load_or_build_vectorstore()
    logger.info("RAG pipeline ready.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Production RAG Chatbot",
    description="Resume-worthy RAG chatbot with evaluation & observability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

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


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "vectorstore_docs": rag.doc_count() if rag else 0}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Standard (non-streaming) chat endpoint."""
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
    """Streaming chat endpoint — tokens arrive in real time."""
    if not rag:
        raise HTTPException(503, "RAG pipeline not ready")

    async def token_generator() -> AsyncGenerator[str, None]:
        async for token in rag.stream_query(req.question):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or TXT document and index it."""
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
    """Run RAGAS evaluation on built-in test set and return scores."""
    scores = rag.evaluate()
    return {"scores": scores, "status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
