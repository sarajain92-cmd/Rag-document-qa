import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
PROMPT_TEMPLATE = """You are a helpful assistant. Answer
the question using ONLY the context below. If the answer
isn't in the context, say "I don't have enough
information in the document to answer that."
Context:
{context}
Question:
{question}
Answer:"""
def generate_answer(question: str,
 context_chunks: list[str]) -> str:
 context = "\n\n".join(context_chunks)
 prompt = PROMPT_TEMPLATE.format(
 context=context, question=question
 )
 response = client.chat.completions.create(
 model="openai/gpt-oss-20b",
 messages=[{"role": "user", "content": prompt}],
 temperature=0.2,
 max_tokens=500,
 )
 return response.choices[0].message.content
