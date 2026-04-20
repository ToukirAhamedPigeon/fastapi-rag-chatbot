import os
from groq import Groq
from dotenv import load_dotenv
from app.services.chroma_store import ChromaStore

load_dotenv()

class RAGChromaEngine:
    def __init__(self):
        print("🔄 Initializing Chroma RAG Engine...")
        self.store = ChromaStore()
        
        # Try to load existing collection
        try:
            self.store.collection = self.store.client.get_collection("documents")
            print("✅ Loaded existing ChromaDB collection")
        except:
            print("⚠️ No collection found. Run load_to_chroma.py first")
        
        # Groq client
        api_key = os.getenv('GROQ_API_KEY')
        self.groq_client = Groq(api_key=api_key) if api_key else None
        print("✅ RAG Engine ready")
    
    def ask(self, query: str) -> dict:
        # Extract price filter from query
        price_max = None
        if 'under $' in query or 'under' in query:
            import re
            match = re.search(r'under\s*\$\s*(\d+)', query)
            if match:
                price_max = int(match.group(1))
                print(f'  Filtering products under ${price_max}')
        
        # Search
        results = self.store.search(query, limit=8, price_max=price_max)
        
        if not results or not results['documents'] or not results['documents'][0]:
            return {"answer": "I couldn't find any relevant information.", "sources": []}
        
        # Build context with price information
        context = ""
        sources = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            price = metadata.get('price', 0)
            context += f"\nProduct: {metadata['title']}\nCategory: {metadata['category']}\nPrice: ${price}\nDescription: {doc[:200]}...\n"
            sources.append(metadata['title'])
        
        # Improved prompt
        prompt = f"""You are a helpful e-commerce assistant. Answer based ONLY on the context below.

CONTEXT:
{context}

QUESTION: {query}

IMPORTANT RULES:
1. If question asks for products under a specific price, ONLY include products with price less than that amount
2. If question asks for price of a specific product (like "price of Jeans"), show that product's price even if it's above any threshold
3. If question asks generally about a category (like "tell me about books"), show products from that category
4. Show product name and exact price for each product
5. Be concise and accurate

ANSWER:"""
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a precise e-commerce assistant. Only answer based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more accurate responses
                max_tokens=400
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error: {str(e)}"
        
        return {"answer": answer, "sources": list(set(sources))}