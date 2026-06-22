from src.document_loader import load_document
from src.chunking import chunk_text
from src.embeddings import embed_texts
from src.vector_store import build_index, load_index, search_index

# 1. Load document
text = load_document("data/uploads/sample.txt")

# 2. Create chunks
chunks = chunk_text(text)

print("Total chunks:", len(chunks))

# 3. Create embeddings
embeddings = embed_texts(chunks)

# 4. Build FAISS index
build_index(embeddings, chunks)

# 5. Load index
index, saved_chunks = load_index()

# 6. Query
q_emb = embed_texts(["what is this document about?"])

# 7. Search
results = search_index(q_emb, index, saved_chunks, top_k=2)

print("Results:", results)
from src.llm import generate_answer
# results from Day 2 search
answer = generate_answer("what is this document about?", results)
print(answer)