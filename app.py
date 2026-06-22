import streamlit as st
import os

from src.document_loader import load_document
from src.chunking import chunk_text
from src.embeddings import embed_texts
from src.vector_store import build_index, load_index, search_index
from src.llm import generate_answer

st.set_page_config(page_title="RAG Document Q&A", layout="wide")
st.title("RAG-based Document Q&A System")

os.makedirs("data/uploads", exist_ok=True)
if st.sidebar.button("Clear index and start over"):
    for path in ["vector_db/index.faiss", "vector_db/chunks.pkl"]:
        if os.path.exists(path):
            os.remove(path)

    st.session_state["chat_history"] = []
    st.sidebar.success("Index cleared successfully.")
# ---------- Sidebar ----------
uploaded_files = st.file_uploader(
    "Choose PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Process documents"):
    all_chunks = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join("data/uploads", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"Reading {uploaded_file.name}..."):
            try:
                text = load_document(file_path)

                if not text.strip():
                    st.warning(
                        f"{uploaded_file.name}: No readable text found (maybe scanned PDF)."
                    )
                    continue

            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {e}")
                continue

        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    if all_chunks:
        with st.spinner("Generating embeddings..."):
            embeddings = embed_texts(all_chunks)

        with st.spinner("Building index..."):
            build_index(embeddings, all_chunks)

        st.success(
            f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} files."
        )


# ---------- Main Chat ----------
st.subheader("Ask a question about your document")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

question = st.text_input("Your question:")

if st.button("Ask") and question:
    try:
        index, chunks = load_index()
    except FileNotFoundError:
        st.error("Please upload and process a document first.")
    else:
        with st.spinner("Searching relevant chunks..."):
            q_embedding = embed_texts([question])
            relevant_chunks = search_index(
                q_embedding, index, chunks, top_k=4
            )

        with st.spinner("Generating answer..."):
            answer = generate_answer(question, relevant_chunks)

        st.session_state["chat_history"].append(
            (question, answer, relevant_chunks)
        )

# ---------- Show Chat ----------
for q, a, sources in reversed(st.session_state["chat_history"]):
    st.markdown(f"**Q: {q}**")
    st.markdown(a)

    with st.expander("View source chunks"):
        for i, chunk in enumerate(sources):
            st.markdown(f"**Chunk {i+1}:** {chunk[:300]}...")

    st.divider()
if st.session_state["chat_history"]:
    transcript = "\n\n".join(
        f"Q: {q}\nA: {a}"
        for q, a, _ in st.session_state["chat_history"]
    )

    st.download_button(
        "Download chat as text",
        data=transcript,
        file_name="qa_transcript.txt",
    )
