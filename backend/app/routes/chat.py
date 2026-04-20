from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from app.services.rag_chroma import RAGChromaEngine

router = APIRouter()
rag_engine = RAGChromaEngine()

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    conversation_id: Optional[str] = None

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = rag_engine.ask(request.message)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            conversation_id=request.conversation_id
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    return {"status": "healthy", "service": "RAG Chatbot with ChromaDB"}