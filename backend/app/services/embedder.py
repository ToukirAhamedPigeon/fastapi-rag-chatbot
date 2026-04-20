from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import time

class Embedder:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = 768  # This should be 768
        print(f"✅ Model loaded. Embedding dimension: {self.dimension}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            print(f"   Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            embeddings = self.model.encode(batch)
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks and add to them"""
        texts = [chunk['content'] for chunk in chunks]
        
        print(f"🔄 Generating embeddings for {len(texts)} chunks...")
        start_time = time.time()
        
        embeddings = self.embed_batch(texts)
        
        elapsed = time.time() - start_time
        print(f"✅ Embeddings generated in {elapsed:.2f} seconds")
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
        
        return chunks

# Test
if __name__ == "__main__":
    from data_loader import DataLoader
    from chunker import TextChunker
    
    # Load and chunk
    loader = DataLoader("data/raw/")
    docs = loader.load_all()
    
    chunker = TextChunker()
    chunks = chunker.chunk_documents(docs)
    
    # Generate embeddings
    embedder = Embedder()
    chunks_with_embeddings = embedder.embed_chunks(chunks[:5])  # Test with 5 chunks
    
    print(f"\n✅ Sample embedding dimension: {len(chunks_with_embeddings[0]['embedding'])}")