import sys
sys.path.append('D:/fastapi-rag-chatbot/backend')
import pandas as pd
from app.services.chroma_store import ChromaStore

# Load data
df = pd.read_csv('D:/fastapi-rag-chatbot/data/raw/ecommerce_products.csv')
print(f'Loaded {len(df)} products')

# Create documents
documents = []
for _, row in df.iterrows():
    content = f"Title: {row['title']}\nCategory: {row['category']}\nDescription: {row['description']}\nPrice: ${row['price']}\nBrand: {row['brand']}"
    documents.append({
        'title': row['title'],
        'content': content,
        'metadata': {
            'category': row['category'],
            'price': float(row['price']),
            'brand': row['brand']
        }
    })

# Add to ChromaDB
store = ChromaStore()
store.create_collection()
store.add_documents(documents)

print('✅ Data loaded to ChromaDB!')