"""
Document ingestion and chunking module.
Supports PDF and TXT files. Loads multiple files at once and splits them
into overlapping chunks ready for embedding.
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Load one or more PDF / TXT files into LangChain Document objects.

    Args:
        file_paths: Absolute or relative paths to documents.

    Returns:
        Flat list of Document objects (one per page for PDFs, one per file for TXT).

    Raises:
        ValueError: If a file type is not supported.
        FileNotFoundError: If a file does not exist.
    """
    all_docs: List[Document] = []

    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path_str}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
            )

        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")

        docs = loader.load()
        # Tag every chunk with the original filename for traceability
        for doc in docs:
            doc.metadata["file_name"] = path.name
        all_docs.extend(docs)

    return all_docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into overlapping text chunks.

    Args:
        documents:    Loaded Document objects.
        chunk_size:   Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of smaller Document chunks preserving original metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
