import streamlit as st
from groq import Groq

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    raise ValueError("❌ GROQ_API_KEY missing in Streamlit Secrets")

client = Groq(api_key=api_key)

PROMPT_TEMPLATE = """You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information".

Context:
{context}

Question:
{question}
"""

def generate_answer(question: str, context_chunks: list) -> str:
    # ✅ argument order fix: question pehle, chunks baad mein
    context = "\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",  # ✅ current working model
        max_tokens=500,
        temperature=0.2,
    )
    return response.choices[0].message.content