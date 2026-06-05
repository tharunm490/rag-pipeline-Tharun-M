"""
Embedding model factory.
Uses the lightweight HuggingFace sentence-transformer model (all-MiniLM-L6-v2)
which runs locally without any API key — fast, free, and 384-dimensional.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFace embedding model instance.
    The model is downloaded on first call and cached in memory afterwards.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
