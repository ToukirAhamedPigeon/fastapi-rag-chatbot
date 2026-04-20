import os
import psycopg2
import json
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

class RAGEngineSimple:
    def __init__(self):
        print("🔄 Initializing RAG Engine...")
        
        # Groq client
        api_key = os.getenv('GROQ_API_KEY')
        self.groq_client = Groq(api_key=api_key) if api_key else None
        
        # Embedding model - use same as database (384 dimensions)
        print("🔄 Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Changed from all-mpnet-base-v2
        print("✅ Embedding model loaded")
        
        # Database connection
        self.db_url = os.getenv('DATABASE_URL')
        print("✅ RAG Engine ready")
    
    def get_connection(self):
        return psycopg2.connect(self.db_url)
    
    def search_documents(self, query: str, limit: int = 10):
        """Search for relevant documents with price filtering"""
        
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        conn = self.get_connection()
        cur = conn.cursor()
        
        # Convert embedding to PostgreSQL vector format
        emb_str = '[' + ','.join(str(x) for x in query_embedding) + ']'
        
        # Check if query asks for price range
        import re
        price_match = re.search(r'under\s*\$\s*(\d+)', query.lower())
        max_price = None
        if price_match:
            max_price = int(price_match.group(1))
            print(f'  Filtering products under ${max_price}')
        
        # Build query with price filter if needed
        if max_price:
            cur.execute('''
                SELECT title, content, metadata, 
                    1 - (embedding <-> %s::vector) as similarity
                FROM documents
                WHERE (metadata->>'price')::float < %s
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            ''', (emb_str, max_price, emb_str, limit))
        else:
            cur.execute('''
                SELECT title, content, metadata, 
                    1 - (embedding <-> %s::vector) as similarity
                FROM documents
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            ''', (emb_str, emb_str, limit))
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        # Filter and format results
        filtered = []
        for r in results:
            if r[3] and r[3] > 0.15:
                filtered.append({
                    'title': r[0],
                    'content': r[1],
                    'metadata': r[2],
                    'similarity': r[3]
                })
        
        return filtered
    
    def generate_answer(self, query: str, context_docs: list) -> str:
        """Generate answer using Groq LLM"""
        if not self.groq_client:
            return "Groq API key not configured."
        
        # Build context
        context_parts = []
        for doc in context_docs:
            context_parts.append(f"Source: {doc['title']}\nContent: {doc['content'][:500]}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""You are a helpful e-commerce assistant. Answer based ONLY on the context below.

CONTEXT:
{context}

QUESTION: {query}

RULES:
1. Answer concisely and accurately
2. If the answer is not in the context, say "I don't have information about that"
3. Include product names and prices if available

ANSWER:"""
        
        try:
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
            return f"Error: {str(e)}"
    
    def ask(self, query: str) -> dict:
        """Main RAG pipeline"""
        relevant_docs = self.search_documents(query)
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find any relevant information.",
                "sources": []
            }
        
        answer = self.generate_answer(query, relevant_docs)
        sources = list(set([doc['title'] for doc in relevant_docs]))
        
        return {
            "answer": answer,
            "sources": sources
        }