from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat

app = FastAPI(
    title="RAG Chatbot API",
    description="AI-powered chatbot with RAG architecture",
    version="1.0.0"
)

# CORS middleware (allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(chat.router)

@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API is running",
        "endpoints": ["/chat", "/health", "/stats"],
        "docs": "/docs"
    }