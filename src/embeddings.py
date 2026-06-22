from sentence_transformers import SentenceTransformer
import numpy as np
_model = None # loaded once, reused (slow to load each time)
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model
def embed_texts(texts: list[str]) -> np.ndarray:
    """Converts a list of text chunks into embedding
    vectors. Returns a NumPy array of shape
    (num_chunks, 384) -- 384 numbers per chunk."""
    model = get_model()
    embeddings = model.encode(
       texts,
       show_progress_bar=False,
        convert_to_numpy=True,
 )
    return embeddings.astype("float32")
