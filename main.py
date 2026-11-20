import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
from typing import List

app = FastAPI(title="Jarvis Personal AI API", version="1.0")

# Simple in-memory memory store (persist to Postgres later)
MEMORY_STORE: List[str] = []

class PromptRequest(BaseModel):
    prompt: str
    use_web: bool = False

class MemoryItem(BaseModel):
    note: str

@app.get("/")
async def home():
    return {"status": "online", "owner": "Casanova (Jarvis)"}

@app.post("/chat")
async def chat_with_brain(req: PromptRequest):
    # 1) optional web search context
    web_context = ""
    if req.use_web:
        try:
            q = requests.utils.requote_uri(req.prompt)
            r = requests.get(f"https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1")
            data = r.json()
            abstract = data.get("AbstractText", "")
            if abstract:
                web_context = f"Web search context:\n{abstract}\n\n"
        except Exception:
            web_context = "(web search failed)\n\n"

    # 2) memory context (last 10 items)
    memories_str = "\n".join(MEMORY_STORE[-10:])
    full_prompt = f"User Memories:\n{memories_str}\n\n{web_context}User Request: {req.prompt}"

    # 3) call Gemini (requires GEMINI_API_KEY in env)
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured in environment variables.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={gemini_api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "content": [{"type": "text", "text": full_prompt}]
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        ai_json = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model request failed: {e}")

    # parse response (best-effort)
    reply = ai_json
    try:
        # support different response shapes; prefer candidates/...parts
        if isinstance(ai_json, dict):
            # new style: "candidates" -> content -> parts
            c = ai_json.get("candidates")
            if c and isinstance(c, list):
                reply = c[0]["content"][0]["text"]
            else:
                # try other path
                reply = ai_json.get("output", ai_json)
    except Exception:
        reply = ai_json

    return {"response": reply, "memory_count": len(MEMORY_STORE)}

@app.post("/memory")
async def save_memory(item: MemoryItem):
    MEMORY_STORE.append(item.note)
    return {"status": "saved", "note": item.note, "memory_count": len(MEMORY_STORE)}

@app.get("/memory")
async def get_memories():
    return {"memories": MEMORY_STORE}
