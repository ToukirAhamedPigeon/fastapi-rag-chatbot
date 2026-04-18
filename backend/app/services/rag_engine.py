import os
import asyncpg
import json
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

class RAGEngine:
    def __init__(self):
        print("🔄 Initializing RAG Engine...")
        
        # Groq client
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("⚠️ Warning: GROQ_API_KEY not found in .env")
        self.groq_client = Groq(api_key=api_key) if api_key else None
        
        # Embedding model
        print("🔄 Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        
        # Database URL
        self.db_url = os.getenv('DATABASE_URL')
        print("✅ RAG Engine ready")
    
    async def search_documents(self, query: str, limit: int = 5):
        """Search for relevant documents using vector similarity"""
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        embedding_vector = '[' + ','.join(str(x) for x in query_embedding) + ']'
        
        # Search in database
        conn = await asyncpg.connect(self.db_url)
        
        results = await conn.fetch('''
            SELECT title, content, metadata, 
                   1 - (embedding <=> $1::vector) as similarity
            FROM documents
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        ''', embedding_vector, limit)
        
        await conn.close()
        
        return [dict(r) for r in results]
    
    async def generate_answer(self, query: str, context_docs: list) -> str:
        """Generate answer using Groq LLM"""
        if not self.groq_client:
            return "Groq API key not configured. Please add GROQ_API_KEY to .env file."
        
        # Build context
        context_parts = []
        for doc in context_docs:
            context_parts.append(f"Source: {doc['title']}\nContent: {doc['content'][:800]}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Prompt template
        prompt = f"""You are a helpful e-commerce assistant. Answer the question based ONLY on the context below.

CONTEXT:
{context}

QUESTION: {query}

RULES:
1. Answer concisely and accurately
2. If the answer is not in the context, say "I don't have information about that"
3. Include product names and prices if available
4. Be friendly and helpful

ANSWER:"""
        
        try:
            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful e-commerce assistant. Answer based on context only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    async def ask(self, query: str) -> dict:
        """Main RAG pipeline"""
        # Step 1: Search relevant documents
        relevant_docs = await self.search_documents(query)
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find any relevant information in our knowledge base.",
                "sources": []
            }
        
        # Step 2: Generate answer
        answer = await self.generate_answer(query, relevant_docs)
        
        # Step 3: Extract sources (unique)
        sources = list(set([doc['title'] for doc in relevant_docs]))
        
        return {
            "answer": answer,
            "sources": sources,
            "similarity_scores": [round(doc['similarity'], 3) for doc in relevant_docs]
        }

# Singleton instance
_rag_engine = None

async def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine