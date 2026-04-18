import asyncio
import sys
sys.path.append('D:/fastapi-rag-chatbot/backend')

async def test():
    from app.services.rag_engine import RAGEngine
    
    print("🔧 Initializing RAG Engine...")
    rag = RAGEngine()
    
    print("\n" + "="*50)
    print("📝 Test Question 1: Product under $50")
    print("="*50)
    result = await rag.ask("What products are available under $50?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Similarity Scores: {result.get('similarity_scores', [])}")
    
    print("\n" + "="*50)
    print("📝 Test Question 2: Electronics")
    print("="*50)
    result = await rag.ask("Tell me about electronic products")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources'][:3]}")

if __name__ == "__main__":
    asyncio.run(test())