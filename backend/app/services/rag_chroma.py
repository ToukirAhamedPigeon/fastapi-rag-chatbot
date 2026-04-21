import os
import re
from groq import Groq
from dotenv import load_dotenv
import urllib.request
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

class RAGChromaEngine:
    def __init__(self):
        try:
            print("Downloading ChromaDB from R2...")
            urllib.request.urlretrieve(
                os.getenv('R2_PUBLIC_URL'), 
                '/tmp/chroma_db.zip'
            )
            import zipfile
            with zipfile.ZipFile('/tmp/chroma_db.zip', 'r') as zip_ref:
                zip_ref.extractall('/tmp/chroma_db')
            print("✅ ChromaDB downloaded")
        except Exception as e:
            print(f"⚠️ Could not download ChromaDB: {e}")
            print("🔄 Initializing Chroma RAG Engine...")
        
        # ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # HuggingFace embedding function (sentence-transformers ছাড়া)
        self.embedding_fn = embedding_functions.HuggingFaceEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            api_key=os.getenv('HF_API_KEY')  # Optional, for rate limiting
        )
        
        # Try to get existing collection
        try:
            self.collection = self.client.get_collection("documents")
            print("✅ Loaded existing ChromaDB collection")
        except:
            print("⚠️ No collection found. Please run load_to_chroma.py first")
            self.collection = None
        
        # Groq client
        api_key = os.getenv('GROQ_API_KEY')
        self.groq_client = Groq(api_key=api_key) if api_key else None
        print("✅ RAG Engine ready")
    
    def ask(self, query: str) -> dict:
        if not self.collection:
            return {"answer": "Database not ready. Please run load_to_chroma.py", "sources": []}
        
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
            return {"answer": "I couldn't find any relevant information.", "sources": []}
        
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