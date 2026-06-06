"""
Streamlit frontend for the RAG Pipeline.
Talks to the FastAPI backend at BACKEND_URL.
"""

import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "https://rag-pipeline-tharun-m.onrender.com")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Document Q&A")
st.caption("Upload your documents and ask questions — answers are grounded in your files.")

# ---------------------------------------------------------------------------
# Sidebar — file upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Upload Documents")

    collection_name = st.text_input(
        "Collection name",
        value="rag_documents",
        help="Give this set of documents a name. Use the same name when querying.",
    )

    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    # top_k slider — controls how many chunks are retrieved
    top_k = st.slider(
        "Chunks to retrieve (top_k)",
        min_value=2,
        max_value=20,
        value=10,
        help="Higher = searches more pages. Increase if answers are missing context.",
    )

    if st.button("⬆️ Ingest Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one file.")
        else:
            with st.spinner("Ingesting documents... (if this is the first request, Render may take 30s to wake up)"):
                files_payload = [
                    ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                    for f in uploaded_files
                ]
                data = {"collection_name": collection_name}
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/upload",
                        files=files_payload,
                        data=data,
                        timeout=180,  # increased timeout for cold start
                    )
                    if resp.ok:
                        result = resp.json()
                        st.success(
                            f"✅ {result['message']}\n\n"
                            f"**Chunks stored:** {result['total_chunks']}\n\n"
                            f"**Files:** {', '.join(result['files_processed'])}"
                        )
                        st.session_state["collection_name"] = collection_name
                    else:
                        try:
                            detail = resp.json().get("detail", resp.text)
                        except Exception:
                            detail = resp.text
                        st.error(f"Upload failed: {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is the server running?")
                except requests.exceptions.Timeout:
                    st.warning("⏳ Request timed out — Render backend may be waking up. Please wait 30 seconds and try again.")
                except Exception as e:
                    st.warning("⏳ Backend may be waking up (Render cold start). Please wait 30 seconds and try again.")

    st.divider()
    st.markdown("**Active collection:** `" + st.session_state.get("collection_name", collection_name) + "`")
    st.caption("ℹ️ First request may take ~30s if backend is sleeping (Render free tier).")

# ---------------------------------------------------------------------------
# Main — Q&A
# ---------------------------------------------------------------------------
active_collection = st.session_state.get("collection_name", collection_name)

st.subheader("💬 Ask a Question")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    page_info = f", page {src['page']}" if src.get("page") is not None else ""
                    st.markdown(f"**{src['file_name']}{page_info}**")
                    st.caption(src["snippet"])

# Input
if user_question := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/query",
                    json={
                        "question": user_question,
                        "collection_name": active_collection,
                        "top_k": top_k,  # uses slider value
                    },
                    timeout=120,
                )
                if resp.ok:
                    data = resp.json()
                    answer  = data["answer"]
                    sources = data["sources"]

                    st.markdown(answer)
                    if sources:
                        with st.expander("📎 Sources"):
                            for src in sources:
                                page_info = f", page {src['page']}" if src.get("page") is not None else ""
                                st.markdown(f"**{src['file_name']}{page_info}**")
                                st.caption(src["snippet"])

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    try:
                        err = resp.json().get("detail", resp.text)
                    except Exception:
                        err = resp.text
                    st.error(f"Error: {err}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Is the server running?")
            except requests.exceptions.Timeout:
                st.warning("⏳ Request timed out — please try again.")
            except Exception as e:
                st.warning("⏳ Backend may be waking up. Please wait 30 seconds and try again.")