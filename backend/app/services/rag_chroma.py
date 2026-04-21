import os
import re
from groq import Groq
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

class RAGChromaEngine:
    def __init__(self):
        print("🔄 Initializing Chroma RAG Engine...")
        
        # ChromaDB client with persistent path
        # Render free tier uses /tmp for temporary storage
        persist_path = "/tmp/chroma_db"
        os.makedirs(persist_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_path)
        print(f"✅ ChromaDB client initialized at {persist_path}")
        
        # HuggingFace embedding function
        self.embedding_fn = embedding_functions.HuggingFaceEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            api_key=os.getenv('HF_API_KEY')
        )
        
        # Try to get existing collection or create empty one
        try:
            self.collection = self.client.get_collection("documents")
            print("✅ Loaded existing ChromaDB collection")
        except:
            # Create empty collection - data will need to be loaded
            self.collection = self.client.create_collection(
                name="documents",
                embedding_function=self.embedding_fn
            )
            print("⚠️ Created new empty ChromaDB collection. Please load data via API.")
            print("   Use POST /load-data endpoint to populate the database.")
        
        # Groq client
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("⚠️ GROQ_API_KEY not found in environment variables")
        self.groq_client = Groq(api_key=api_key) if api_key else None
        print("✅ RAG Engine ready")
    
    def ask(self, query: str) -> dict:
        if not self.collection:
            return {"answer": "Database not ready. Please load data first.", "sources": []}
        
        # Extract price filter
        price_max = None
        match = re.search(r'under\s*\$\s*(\d+)', query.lower())
        if match:
            price_max = int(match.group(1))
            print(f'  Filtering products under ${price_max}')
        
        # Search
        where_filter = {}
        if price_max:
            where_filter = {"price": {"$lt": price_max}}
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=5,
                where=where_filter if where_filter else None
            )
        except Exception as e:
            return {"answer": f"Search error: {str(e)}", "sources": []}
        
        if not results or not results['documents'] or not results['documents'][0]:
            return {"answer": "I couldn't find any relevant information. Please load product data first.", "sources": []}
        
        # Build context
        context = ""
        sources = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            price = metadata.get('price', 'N/A')
            context += f"\nProduct: {metadata['title']}\nPrice: ${price}\nDescription: {doc[:300]}...\n"
            sources.append(metadata['title'])
        
        # Generate answer with Groq
        prompt = f"""You are a helpful e-commerce assistant. Answer based ONLY on the context.

CONTEXT:
{context}

QUESTION: {query}

RULES:
1. List products with prices
2. If price filter was asked, only include matching products
3. Be concise

ANSWER:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error: {str(e)}"
        
        return {"answer": answer, "sources": list(set(sources))}