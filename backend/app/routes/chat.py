from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from app.services.rag_engine import get_rag_engine

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    conversation_id: Optional[str] = None

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - RAG powered"""
    try:
        rag = await get_rag_engine()
        result = await rag.ask(request.message)
        
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
    return {"status": "healthy", "service": "RAG Chatbot"}

@router.get("/stats")
async def stats():
    """Get database statistics"""
    try:
        import asyncpg
        import os
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        count = await conn.fetchval('SELECT COUNT(*) FROM documents')
        await conn.close()
        return {"total_documents": count}
    except Exception as e:
        return {"error": str(e)}