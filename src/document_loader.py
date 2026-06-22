import os
from pypdf import PdfReader
def load_pdf(file_path: str) -> str:
    """Reads a PDF file page by page and returns
    its full text content as one string."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
           text += page_text + "\n"
    return text
def load_txt(file_path: str) -> str:
    """Reads a plain text file and returns its content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
def load_document(file_path: str) -> str:
    """Detects the file type from its extension
    and calls the right loader function."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")