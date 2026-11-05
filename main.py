from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

app = FastAPI(title="Gemini Chat API")

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

SYSTEM_PROMPT = "You are a concise, highly technical AI mentor. You give direct answers without fluff."
MAX_HISTORY = 6 

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        truncated_history = req.history[-MAX_HISTORY:] if len(req.history) > MAX_HISTORY else req.history

        # Convert simple history dicts to Content types
        formatted_history = []
        for msg in truncated_history:
            formatted_history.append(
                types.Content(role=msg["role"], parts=[types.Part.from_text(text=msg["parts"][0])])
            )
            
        # Initialize chat
        if formatted_history:
            chat = client.chats.create(
                model='gemini-3.6-flash', 
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                history=formatted_history
            )
        else:
            chat = client.chats.create(
                model='gemini-3.6-flash', 
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
            )

        # Get response
        response = chat.send_message(req.message)

        # Update history array to return to the frontend
        updated_history = truncated_history + [
            {"role": "user", "parts": [req.message]},
            {"role": "model", "parts": [response.text]}
        ]
        
        return {"reply": response.text, "history": updated_history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))