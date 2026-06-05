"""
Vector store module wrapping ChromaDB.
Handles persistence, document ingestion, and semantic similarity search.
Collection names are derived from the file name so each upload gets its
own isolated namespace inside the shared ChromaDB directory.
"""

import re
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.embeddings import get_embeddings

CHROMA_DIR = "./chroma_db"


def _sanitize_collection(name: str) -> str:
    """Convert a filename into a valid Chroma collection name."""
    base = Path(name).stem  # strip extension
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", base)
    # Chroma requires 3-63 chars, starts with letter/digit
    sanitized = sanitized[:60] or "default_collection"
    if not sanitized[0].isalnum():
        sanitized = "c_" + sanitized
    return sanitized


def get_vectorstore(collection_name: str = "rag_documents") -> Chroma:
    """
    Return a persistent ChromaDB vectorstore for the given collection.
    Creates the collection if it does not exist yet.
    """
    return Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )


def ingest_chunks(
    chunks: List[Document],
    collection_name: str = "rag_documents",
) -> int:
    """
    Embed and store document chunks in ChromaDB.

    Args:
        chunks:          Pre-split Document objects.
        collection_name: Target Chroma collection.

    Returns:
        Number of chunks successfully stored.
    """
    vectorstore = get_vectorstore(collection_name)
    vectorstore.add_documents(chunks)
    return len(chunks)


def retrieve_context(
    query: str,
    collection_name: str = "rag_documents",
    k: int = 4,
) -> List[Document]:
    """
    Perform a semantic similarity search and return the top-k chunks.

    Args:
        query:           User question.
        collection_name: Chroma collection to search.
        k:               Number of chunks to return.

    Returns:
        List of the most relevant Document chunks.
    """
    vectorstore = get_vectorstore(collection_name)
    return vectorstore.similarity_search(query, k=k)


def collection_name_from_file(file_name: str) -> str:
    """Public helper: derive a Chroma collection name from a file name."""
    return _sanitize_collection(file_name)
