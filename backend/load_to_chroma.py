import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import json

def load_data_to_chroma():
    print("Loading data to ChromaDB...")
    
    # Load CSV data
    df = pd.read_csv('D:/fastapi-rag-chatbot/data/raw/ecommerce_products.csv')
    print(f"Loaded {len(df)} products")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="/tmp/chroma_db")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Delete existing collection if exists
    try:
        client.delete_collection("documents")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name="documents",
        embedding_function=embedding_fn
    )
    
    # Add documents
    ids = []
    documents = []
    metadatas = []
    
    for idx, row in df.iterrows():
        content = f"Title: {row['title']}\nCategory: {row['category']}\nDescription: {row['description']}\nPrice: ${row['price']}\nBrand: {row['brand']}"
        
        ids.append(str(idx))
        documents.append(content)
        metadatas.append({
            'title': row['title'],
            'category': row['category'],
            'price': float(row['price']),
            'brand': row['brand']
        })
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"✅ Added {len(ids)} documents to ChromaDB")
    return True

if __name__ == "__main__":
    load_data_to_chroma()