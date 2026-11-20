import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(title="Casanova Personal AI API", version="1.0")

MEMORY_STORE = []

class PromptRequest(BaseModel):
    prompt: str
    use_web: bool = False

@app.get("/")
def home():
    return {"status": "online", "owner": "Casanova"}

@app.post("/chat")
def chat_with_brain(req: PromptRequest):
    web_context = ""
    if req.use_web:
        try:
            search_url = f"https://api.duckduckgo.com/?q={req.prompt}&format=json"
            res = requests.get(search_url).json()
            abstract = res.get("AbstractText", "")
            if abstract:
                web_context = f"Web search context: {abstract}\n\n"
        except Exception:
            web_context = "Web search failed.\n\n"

    memories_str = "\n".join(MEMORY_STORE[-5:])
    full_prompt = f"User Memories:\n{memories_str}\n\n{web_context}User Request: {req.prompt}"

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={gemini_api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}]
    }

    ai_response = requests.post(url, headers=headers, json=payload).json()
    
    try:
        reply = ai_response["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        reply = f"Error parsing model response: {ai_response}"

    return {"response": reply, "memory_count": len(MEMORY_STORE)}

@app.post("/memory")
def save_memory(note: str):
    MEMORY_STORE.append(note)
    return {"status": "saved", "note": note}

@app.get("/memory")
def get_memories():
    return {"memories": MEMORY_STORE}
