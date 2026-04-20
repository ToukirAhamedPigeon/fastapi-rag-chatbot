import sys
sys.path.append('D:/fastapi-rag-chatbot/backend')
import pandas as pd
from sentence_transformers import SentenceTransformer
import psycopg2
import os
import json
from dotenv import load_dotenv

load_dotenv()

print('Loading embedding model...')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print('Loading data...')
df = pd.read_csv('D:/fastapi-rag-chatbot/data/raw/ecommerce_products.csv')
print(f'Loaded {len(df)} products')

# Create texts for embedding
texts = []
for _, row in df.iterrows():
    text = f"Title: {row['title']}\nCategory: {row['category']}\nDescription: {row['description']}\nPrice: {row['price']}"
    texts.append(text)

print('Generating embeddings...')
embeddings = embedder.encode(texts, show_progress_bar=True)

print('Connecting to database...')
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

print('Inserting data...')
for idx, row in df.iterrows():
    embedding_list = embeddings[idx].tolist()
    metadata = {
        'category': row['category'],
        'price': float(row['price']),
        'brand': row['brand']
    }
    
    cur.execute('''
        INSERT INTO documents (title, content, embedding, metadata)
        VALUES (%s, %s, %s, %s)
    ''', (row['title'], texts[idx], embedding_list, json.dumps(metadata)))
    
    if (idx + 1) % 100 == 0:
        conn.commit()
        print(f'  Inserted {idx + 1}/{len(df)} documents')

conn.commit()
print(f'✅ Inserted {len(df)} documents')
cur.close()
conn.close()