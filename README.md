# RAG Pipeline

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline built with:

| Layer | Technology |
|---|---|
| Document loading | LangChain `PyPDFLoader` / `TextLoader` |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embeddings | `all-MiniLM-L6-v2` (local, no API key) |
| Vector store | ChromaDB (persistent) |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containers | Docker + docker-compose |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        User                               │
└───────────────────────┬──────────────────────────────────┘
                        │  Upload docs / Ask question
                        ▼
┌──────────────────────────────────────────────────────────┐
│              Streamlit Frontend  (:8501)                  │
└───────────────────────┬──────────────────────────────────┘
                        │  HTTP (REST)
                        ▼
┌──────────────────────────────────────────────────────────┐
│              FastAPI Backend  (:8000)                     │
│                                                           │
│  POST /upload ──► Ingestion Pipeline                      │
│                     1. Load PDF/TXT                       │
│                     2. Chunk (size=1000, overlap=200)     │
│                     3. Embed (all-MiniLM-L6-v2)          │
│                     4. Store → ChromaDB                   │
│                                                           │
│  POST /query  ──► RAG Chain                               │
│                     1. Embed query                        │
│                     2. Retrieve top-k chunks              │
│                     3. Build prompt + context             │
│                     4. Generate → Groq (Llama 3.3 70B)   │
│                     5. Return answer + sources            │
└──────────────┬───────────────────────────────────────────┘
               │
       ┌───────┴────────┐
       │   ChromaDB     │  (persisted to ./chroma_db/)
       └────────────────┘
```

---

## Quick Start (without Docker)

### 1. Clone & install
```bash
git clone https://github.com/tharunm490/rag-pipeline-Tharun-M.git


pip install uv ##if you do not have the uv installed
uv init
uv venv .venv
source .venv/Scripts/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. Set your Groq API key
```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
# Get a free key at https://console.groq.com
```

### 3. Run the FastAPI backend
```bash
python run.py 
# or
uvicorn app.api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 4. Run the Streamlit frontend (separate terminal)
```bash
streamlit run streamlit_app.py
# Opens: http://localhost:8501
```

---

## 🚀 Quick Run (No code needed)

### File Structure
any-folder/
├── .env     #in this file add your api_key           
└── docker-compose.yml  

### Step 1 — Create a `.env` file
Create a new file called `.env` and add your Groq API key:
Get your free key at → https://console.groq.com
and in the .env file it should be like this 
api_key="your_key"
---

### Step 2 — Create a `docker-compose.yml` file
Create a new file called `docker-compose.yml` and paste this:

```yaml
services:
  api:
    image: tharunm490/rag-pipeline-api:latest
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./chroma_db:/app/chroma_db
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 20s

  streamlit:
    image: tharunm490/rag-pipeline-streamlit:latest
    ports:
      - "8501:8501"
    env_file:
      - .env
    environment:
      - BACKEND_URL=http://api:8000
    depends_on:
      api:
        condition: service_healthy
```

---

### Step 3 — Run
```bash
docker-compose up
```
Docker automatically pulls both images from Docker Hub. No cloning, no pip install, nothing else.

---

### Step 4 — Open browser
| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| API Health | http://localhost:8000/health |
| Swagger Docs | http://localhost:8000/docs |

### Stop
```bash
docker-compose down
```
---

## API Reference

### `POST /upload`
Upload one or more PDF or TXT files.

**Form data:**
| Field | Type | Default | Description |
|---|---|---|---|
| `files` | `File[]` | required | PDF or TXT files |
| `collection_name` | `string` | `rag_documents` | ChromaDB collection |

**Response:**
```json
{
  "message": "Successfully ingested 2 file(s).",
  "files_processed": ["report.pdf", "notes.txt"],
  "total_chunks": 312,
  "collection_name": "rag_documents"
}
```

### `POST /query`
Ask a question.

**JSON body:**
```json
{
  "question": "What are the admin roles in Microsoft Fabric?",
  "collection_name": "rag_documents",
  "top_k": 4
}
```

**Response:**
```json
{
  "answer": "Microsoft Fabric has several admin roles...",
  "sources": [
    {
      "file_name": "fabric-admin.pdf",
      "page": 8,
      "snippet": "Microsoft Fabric admin is the management of..."
    }
  ]
}
```

### `GET /health`
Returns `{"status": "ok"}`.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Document loading (TXT, error cases)
- Chunking logic and metadata preservation
- ChromaDB collection name sanitisation
- RAG chain formatting and empty-store handling

---

## Project Structure

```
rag-pipeline/
├── app/
│   ├── api/
│   │   └── main.py          # FastAPI endpoints
│   └── core/
│       ├── ingestion.py     # Load + chunk documents
│       ├── embeddings.py    # HuggingFace embedding model
│       ├── vectorstore.py   # ChromaDB operations
│       ├── llm.py           # Groq LLM
│       └── rag_chain.py     # Retrieval + generation
├── tests/
│   └── test_core.py         # Unit tests
├── streamlit_app.py         # Streamlit UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Design Decisions

**Why `all-MiniLM-L6-v2`?**  
Fast, lightweight (22 MB), runs on CPU, no API key needed. 384-dimensional vectors give excellent semantic similarity for document Q&A tasks.

**Why ChromaDB?**  
Zero-config persistent vector store with a clean Python API. `langchain-chroma` integration is first-class and the `./chroma_db/` directory is easy to mount in Docker.

**Why Groq?**  
Groq provides the fastest Llama 3 inference available (hundreds of tokens/second) via a simple OpenAI-compatible API. Free tier is generous enough for development and demos.

**Chunking strategy**  
`chunk_size=1000, chunk_overlap=200` balances context preservation with retrieval precision. `RecursiveCharacterTextSplitter` tries to split at paragraph → sentence → word boundaries, keeping semantic units intact.

**Collection-per-upload**  
All files in a single `/upload` request share a collection name (user-defined). This lets you have multiple isolated document sets in the same ChromaDB instance.

## 🧪 Testing

This project includes **13 unit tests** covering all core pipeline functions.

### Run the tests
```bash
pytest tests/test_core.py -v
```

### What is unit testing?
Unit testing verifies that each individual function of the application works 
correctly in isolation — without needing a real database, real API calls, or 
real files.

### Test coverage

| Test Class | Function Tested | What it verifies |
|---|---|---|
| `TestChunkDocuments` | `chunk_documents()` | Splitting works correctly, metadata preserved, empty doc handled |
| `TestLoadDocuments` | `load_documents()` | TXT loading, wrong file type raises error, missing file raises error |
| `TestCollectionName` | `collection_name_from_file()` | Special characters sanitised, valid ChromaDB collection name generated |
| `TestFormatContext` | `_format_context()` | Page numbers appear correctly, works without page number |
| `TestAnswerQuery` | `answer_query()` | Returns answer + sources, empty vectorstore handled gracefully |

### Why mocking?
External dependencies like the **Groq LLM** and **ChromaDB** are mocked in tests.
This means:
- ✅ No API key needed to run tests
- ✅ Works fully offline
- ✅ Fast execution (13 tests in ~30 seconds)
- ✅ Reliable — tests don't fail due to third-party service outages
- ✅ Free — no API cost per test run

### Expected output
```
tests/test_core.py::TestChunkDocuments::test_basic_chunking         PASSED
tests/test_core.py::TestChunkDocuments::test_metadata_preserved     PASSED
tests/test_core.py::TestChunkDocuments::test_empty_document         PASSED
tests/test_core.py::TestLoadDocuments::test_unsupported_extension_raises  PASSED
tests/test_core.py::TestLoadDocuments::test_missing_file_raises     PASSED
tests/test_core.py::TestLoadDocuments::test_txt_loading             PASSED
tests/test_core.py::TestCollectionName::test_sanitize_basic         PASSED
tests/test_core.py::TestCollectionName::test_sanitize_special_chars PASSED
tests/test_core.py::TestCollectionName::test_sanitize_leading_non_alnum  PASSED
tests/test_core.py::TestFormatContext::test_format_with_page        PASSED
tests/test_core.py::TestFormatContext::test_format_without_page     PASSED
tests/test_core.py::TestAnswerQuery::test_returns_answer_and_sources PASSED
tests/test_core.py::TestAnswerQuery::test_empty_vectorstore         PASSED

============= 13 passed in 30.56s =============
```