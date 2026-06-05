"""
FastAPI backend for the RAG pipeline.

Endpoints:
  POST /upload   - Upload one or more PDF/TXT files, chunk and embed them.
  POST /query    - Ask a question against the ingested documents.
  GET  /health   - Health check.
  GET  /docs-list - List collections (ingested document groups) in ChromaDB.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from app.core.ingestion import load_documents, chunk_documents
from app.core.vectorstore import ingest_chunks, collection_name_from_file
from app.core.rag_chain import answer_query

load_dotenv()  # picks up GROQ_API_KEY from .env

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Pipeline API",
    description="Upload documents and ask questions. Powered by ChromaDB + Groq.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    collection_name: Optional[str] = "rag_documents"
    top_k: Optional[int] = 4


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


class UploadResponse(BaseModel):
    message: str
    files_processed: List[str]
    total_chunks: int
    collection_name: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RAG Pipeline is running."}


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection_name: Optional[str] = Form(default="rag_documents"),
):
    """
    Upload one or more PDF or TXT files.
    Documents are chunked and embedded into ChromaDB immediately.
    All files in a single request share the same collection_name.
    """
    allowed_types = {"application/pdf", "text/plain"}
    allowed_exts  = {".pdf", ".txt"}

    processed_files = []
    total_chunks = 0
    tmp_dir = tempfile.mkdtemp()

    try:
        saved_paths = []
        for upload in files:
            ext = Path(upload.filename).suffix.lower()
            if ext not in allowed_exts:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {upload.filename}. Use PDF or TXT.",
                )
            dest = Path(tmp_dir) / upload.filename
            with open(dest, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved_paths.append(str(dest))
            processed_files.append(upload.filename)

        # Ingest pipeline
        docs   = load_documents(saved_paths)
        chunks = chunk_documents(docs)
        count  = ingest_chunks(chunks, collection_name=collection_name)
        total_chunks += count

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return UploadResponse(
        message=f"Successfully ingested {len(processed_files)} file(s).",
        files_processed=processed_files,
        total_chunks=total_chunks,
        collection_name=collection_name,
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Ask a question against the ingested documents.
    Returns an AI-generated answer with source citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = answer_query(
        query=request.question,
        collection_name=request.collection_name,
        k=request.top_k,
    )
    return QueryResponse(**result)
