import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import json

class ChromaStore:
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = None
    
    def create_collection(self, name="documents"):
        # Delete existing if exists
        try:
            self.client.delete_collection(name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Collection '{name}' created")
        return self.collection
    
    def add_documents(self, documents):
        """Add documents to ChromaDB"""
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            ids.append(str(i))
            texts.append(doc['content'])
            metadatas.append({
                'title': doc['title'],
                'category': doc['metadata'].get('category', 'General'),
                'price': float(doc['metadata'].get('price', 0)),
                'brand': doc['metadata'].get('brand', '')
            })
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        print(f"✅ Added {len(documents)} documents to ChromaDB")
    
    def search(self, query, limit=5, price_max=None, category=None):
        """Search with filters"""
        where_filter = {}
        
        if price_max:
            where_filter['price'] = {"$lt": price_max}
        if category:
            where_filter['category'] = {"$eq": category}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=where_filter if where_filter else None
        )
        
        return results