import streamlit as st
from groq import Groq

# ✅ safe तरीके से key लेना
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception as e:
    raise ValueError("❌ GROQ_API_KEY missing in Streamlit Secrets")

# ✅ client init
client = Groq(api_key=api_key)

PROMPT_TEMPLATE = """You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information".

Context:
{context}

Question:
{question}
"""

def generate_answer(context, question):
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )

    return response.choices[0].message.content