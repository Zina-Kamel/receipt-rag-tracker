import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-pro")

def answer_query(query: str, contexts: list[str]) -> str:
    context_str = "\n---\n".join(contexts)
    prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"

    response = model.generate_content(prompt)
    return response.text.strip()
