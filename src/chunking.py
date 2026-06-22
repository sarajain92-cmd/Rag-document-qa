def chunk_text(text: str, chunk_size: int = 500,
overlap: int = 50) -> list[str]:
    """Splits text into overlapping chunks of
    approximately chunk_size words each.
    Overlap ensures a sentence that falls right at
    a chunk boundary isn't lost from context."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
       end = start + chunk_size
       chunk = " ".join(words[start:end])
       chunks.append(chunk)
    # move forward, but re-include the last
    # `overlap` words so context isn't cut off
       start = end - overlap
    return chunks