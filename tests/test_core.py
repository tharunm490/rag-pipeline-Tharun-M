"""
Unit tests for the RAG pipeline core modules.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# ingestion tests
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    """Tests for app.core.ingestion.chunk_documents"""

    def test_basic_chunking(self):
        from app.core.ingestion import chunk_documents

        doc = Document(page_content="word " * 300, metadata={"source": "test.txt"})
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 250  # small leeway

    def test_metadata_preserved(self):
        from app.core.ingestion import chunk_documents

        doc = Document(page_content="Hello world. " * 100, metadata={"file_name": "my.pdf", "page": 1})
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata.get("file_name") == "my.pdf"

    def test_empty_document(self):
        from app.core.ingestion import chunk_documents

        chunks = chunk_documents([Document(page_content="", metadata={})])
        # Empty doc should produce zero or one empty chunk — just not crash
        assert isinstance(chunks, list)


class TestLoadDocuments:
    """Tests for app.core.ingestion.load_documents"""

    def test_unsupported_extension_raises(self, tmp_path):
        from app.core.ingestion import load_documents

        bad_file = tmp_path / "doc.docx"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_documents([str(bad_file)])

    def test_missing_file_raises(self):
        from app.core.ingestion import load_documents

        with pytest.raises(FileNotFoundError):
            load_documents(["/nonexistent/path/file.pdf"])

    def test_txt_loading(self, tmp_path):
        from app.core.ingestion import load_documents

        txt = tmp_path / "sample.txt"
        txt.write_text("Hello from a text file.")
        docs = load_documents([str(txt)])
        assert len(docs) >= 1
        assert docs[0].metadata["file_name"] == "sample.txt"
        assert "Hello" in docs[0].page_content


# ---------------------------------------------------------------------------
# vectorstore tests
# ---------------------------------------------------------------------------

class TestCollectionName:
    def test_sanitize_basic(self):
        from app.core.vectorstore import collection_name_from_file

        assert collection_name_from_file("my document.pdf") == "my_document"

    def test_sanitize_special_chars(self):
        from app.core.vectorstore import collection_name_from_file

        result = collection_name_from_file("file!@#$.txt")
        assert all(c.isalnum() or c in "-_" for c in result)

    def test_sanitize_leading_non_alnum(self):
        from app.core.vectorstore import collection_name_from_file

        result = collection_name_from_file("_bad_start.pdf")
        assert result[0].isalnum() or result.startswith("c_")


# ---------------------------------------------------------------------------
# rag_chain tests
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_format_with_page(self):
        from app.core.rag_chain import _format_context

        doc = Document(
            page_content="Some content here.",
            metadata={"file_name": "report.pdf", "page": 5},
        )
        result = _format_context([doc])
        assert "report.pdf" in result
        assert "page 5" in result
        assert "Some content here." in result

    def test_format_without_page(self):
        from app.core.rag_chain import _format_context

        doc = Document(page_content="Text.", metadata={"file_name": "notes.txt"})
        result = _format_context([doc])
        assert "notes.txt" in result
        assert "page" not in result


class TestAnswerQuery:
    @patch("app.core.rag_chain.retrieve_context")
    @patch("app.core.rag_chain.get_llm")
    def test_returns_answer_and_sources(self, mock_get_llm, mock_retrieve):
        from app.core.rag_chain import answer_query

        mock_retrieve.return_value = [
            Document(
                page_content="Fabric admin manages org settings.",
                metadata={"file_name": "fabric.pdf", "page": 1},
            )
        ]
        mock_llm = MagicMock()
        mock_llm.__or__ = MagicMock(return_value=mock_llm)
        mock_get_llm.return_value = mock_llm

        # Patch the full chain instead of mocking chain internals
        with patch("app.core.rag_chain.prompt") as mock_prompt:
            chain_mock = MagicMock()
            chain_mock.invoke.return_value = "Fabric admin is great."
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            result = answer_query("What is Fabric admin?")

        assert "answer" in result
        assert "sources" in result

    @patch("app.core.rag_chain.retrieve_context", return_value=[])
    def test_empty_vectorstore(self, _):
        from app.core.rag_chain import answer_query

        result = answer_query("anything")
        assert "No documents found" in result["answer"]
        assert result["sources"] == []
