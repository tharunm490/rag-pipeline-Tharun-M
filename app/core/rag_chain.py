"""
RAG chain module.
Combines retrieval from ChromaDB with generation via Groq.
Uses a structured prompt that instructs the LLM to stay grounded
in the provided context and cite the source document names.
"""

from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.llm import get_llm
from app.core.vectorstore import retrieve_context


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based \
strictly on the provided context from the user's documents.

Rules:
- Answer ONLY from the context below; do not add outside knowledge.
- If the answer is not in the context, say: "I couldn't find that information \
in the uploaded documents."
- Cite the source document name (file_name from metadata) when relevant.
- Be concise and clear.

Context:
{context}
"""

USER_TEMPLATE = "Question: {question}"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(docs: List[Document]) -> str:
    """Serialise retrieved chunks into a readable context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("file_name", "unknown")
        page = doc.metadata.get("page", "")
        page_info = f", page {page}" if page != "" else ""
        parts.append(
            f"[Source {i}: {source}{page_info}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    collection_name: str = "rag_documents",
    k: int = 4,
) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve → format context → generate answer.

    Args:
        query:           User question.
        collection_name: Chroma collection to search.
        k:               Number of context chunks to retrieve.

    Returns:
        Dictionary with keys:
          - answer (str): Generated answer text.
          - sources (list[dict]): Metadata of the retrieved source chunks.
    """
    # 1. Retrieve relevant chunks
    docs = retrieve_context(query, collection_name=collection_name, k=k)

    if not docs:
        return {
            "answer": "No documents found in the vector store. Please upload documents first.",
            "sources": [],
        }

    # 2. Format context
    context_str = _format_context(docs)

    # 3. Build and invoke the chain
    chain = prompt | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context_str, "question": query})

    # 4. Collect source metadata for the response
    sources = [
        {
            "file_name": doc.metadata.get("file_name", "unknown"),
            "page": doc.metadata.get("page", None),
            "snippet": doc.page_content[:200] + "...",
        }
        for doc in docs
    ]

    return {"answer": answer, "sources": sources}
