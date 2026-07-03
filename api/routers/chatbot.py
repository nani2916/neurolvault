from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))


class ChatRequest(BaseModel):
    question: str
    banker_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources_used: int


@router.post("/ask", response_model=ChatResponse)
async def ask_neurolvault(request: ChatRequest):
    """
    Chatbot endpoint for Salesforce LWC.
    Banker asks question → RAG finds context → Groq answers.
    
    URL: POST /api/chatbot/ask
    """
    try:
        from rag_system.llm_gateway import ask_neurolvault
        answer = ask_neurolvault(request.question)

        return ChatResponse(
            answer=answer,
            sources_used=3
        )

    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def chatbot_health():
    return {"status": "Chatbot endpoint running"}