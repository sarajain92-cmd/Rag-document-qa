import faiss
import numpy as np
import pickle
import os
INDEX_PATH = "vector_db/index.faiss"
CHUNKS_PATH = "vector_db/chunks.pkl"
def build_index(embeddings: np.ndarray, chunks: list[str]):
    """Builds a FAISS index from embeddings and saves
    it to disk, along with the original chunk texts
    (needed later to retrieve the actual text, since
    FAISS only stores numbers, not text)."""
    os.makedirs("vector_db", exist_ok=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    return index
def load_index():
    """Loads a previously saved FAISS index and the
    matching chunk texts."""
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks
def search_index(query_embedding: np.ndarray, index,
    chunks: list[str], top_k: int = 4):
    """ Searches the FAISS index for the top_k chunks
    most similar to the query embedding."""
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0] if i != -1]
    return results