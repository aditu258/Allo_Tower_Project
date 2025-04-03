from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from chat_backend import initialize_chat_system, get_chat_response, reset_chat_history
import uvicorn
import threading
from pathlib import Path
from typing import Dict, Any
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure static files
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Global chat system instance
chat_system = None
chat_system_lock = threading.Lock()

'''@app.on_event("startup")
async def startup_event():
    """Initialize chat system on startup"""
    global chat_system
    with chat_system_lock:
        if chat_system is None:
            print("Initializing chat system...")
            chat_system = initialize_chat_system()
            print("Chat system initialized successfully")'''

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/send_message")
async def send_message(request: Request) -> Dict[str, Any]:
    """Handle chat messages"""
    global chat_system
    try:
        data = await request.json()
        user_message = data.get("message", "").strip()
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Empty message")
        
        if chat_system is None:
            with chat_system_lock:
                if chat_system is None:
                    chat_system = initialize_chat_system()
        
        bot_response = get_chat_response(chat_system, user_message)
        return {
            "user_message": user_message,
            "bot_response": bot_response,
            "timestamp": int(time.time())
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_chat")
async def reset_chat():
    """Reset chat history"""
    global chat_system
    try:
        if chat_system is not None:
            reset_chat_history(chat_system)
        return {"status": "success", "timestamp": int(time.time())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/load_history")
async def load_history():
    """Load chat history"""
    global chat_system
    try:
        if chat_system is None:
            return {"history": []}
        
        from chat_backend import serialize_chat_history
        history = serialize_chat_history(chat_system.chat_history)
        return {
            "history": [{
                **msg,
                "timestamp": int(time.time())
            } for msg in history]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ready" if chat_system else "initializing"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT not set
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True if os.getenv("ENV") == "development" else False,
        workers=1
    )