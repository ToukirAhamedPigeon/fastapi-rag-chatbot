import faiss
import numpy as np
import pickle
import os

class FAISSStore:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        self.documents = []
        self.is_trained = False
    
    def add_documents(self, embeddings, documents):
        """Add documents to FAISS index"""
        embeddings_np = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)  # Normalize for cosine similarity
        self.index.add(embeddings_np)
        self.documents.extend(documents)
        print(f"✅ Added {len(documents)} documents to FAISS")
    
    def search(self, query_embedding, k=5):
        """Search for similar documents"""
        query_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_np)
        
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                results.append({
                    'title': self.documents[idx]['title'],
                    'content': self.documents[idx]['content'],
                    'metadata': self.documents[idx]['metadata'],
                    'similarity': float(distances[0][i])
                })
        return results
    
    def save(self, path='faiss_index.pkl'):
        """Save FAISS index to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'documents': self.documents,
                'dimension': self.dimension
            }, f)
        print(f"✅ Saved FAISS index to {path}")
    
    def load(self, path='faiss_index.pkl'):
        """Load FAISS index from disk"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.index = data['index']
                self.documents = data['documents']
                self.dimension = data['dimension']
            print(f"✅ Loaded FAISS index with {len(self.documents)} documents")
            return True
        return False