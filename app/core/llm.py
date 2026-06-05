"""
LLM module.
Uses Groq's API (llama-3.3-70b-versatile) via the langchain-groq integration.
Groq provides extremely fast inference for Llama 3 models.

Set GROQ_API_KEY in your .env file before running.
"""

from functools import lru_cache

from langchain_groq import ChatGroq


GROQ_MODEL = "llama-3.3-70b-versatile"


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Return a cached ChatGroq LLM instance.
    The GROQ_API_KEY is automatically read from the environment.
    """
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,     # Lower temp = more factual, less hallucination
        max_tokens=1024,
    )
