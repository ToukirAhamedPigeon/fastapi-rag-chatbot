import asyncpg
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# সঠিক ইম্পোর্ট - relative import ব্যবহার করুন
from .data_loader import DataLoader
from .chunker import TextChunker
from .embedder import Embedder
from sentence_transformers import SentenceTransformer

load_dotenv()

class VectorStore:
    def __init__(self):
        self.connection_string = os.getenv('DATABASE_URL')
        self.pool = None
    
    async def connect(self):
        """Create connection pool to Neon PostgreSQL"""
        self.pool = await asyncpg.create_pool(self.connection_string)
        
        async with self.pool.acquire() as conn:
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            print("✅ pgvector extension enabled")
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    embedding VECTOR(384),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            print("✅ Documents table ready")
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_embedding 
                ON documents 
                USING ivfflat (embedding vector_cosine_ops)
            ''')
            print("✅ Vector index ready")
    
    async def insert_documents(self, chunks: List[Dict[str, Any]]):
        """Insert document chunks with embeddings"""
        async with self.pool.acquire() as conn:
            for chunk in chunks:
                # embedding কে vector টাইপে কাস্ট করুন
                embedding_vector = '[' + ','.join(str(float(x)) for x in chunk['embedding']) + ']'
                
                await conn.execute('''
                    INSERT INTO documents (title, content, embedding, metadata)
                    VALUES ($1, $2, $3::vector, $4)
                ''', 
                    chunk['metadata'].get('title', 'Unknown'),
                    chunk['content'],
                    embedding_vector,
                    json.dumps(chunk['metadata'])
                )
        print(f"✅ Inserted {len(chunks)} chunks into database")
    
    async def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity"""
        async with self.pool.acquire() as conn:
            results = await conn.fetch('''
                SELECT 
                    title,
                    content,
                    metadata,
                    1 - (embedding <=> $1::vector) as similarity
                FROM documents
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            ''', query_embedding, limit)
            
            return [dict(r) for r in results]

    # Complete processing pipeline
    async def process_all_data():
        """Complete processing pipeline"""
        # Load
        print("📖 Loading documents...")
        loader = DataLoader("D:/fastapi-rag-chatbot/data/raw/")
        docs = loader.load_all()
        print(f"   Loaded {len(docs)} documents")
        
        # Chunk
        print("\n✂️  Chunking documents...")
        chunker = TextChunker()
        chunks = chunker.chunk_documents(docs)
        print(f"   Created {len(chunks)} chunks")
        
        # Embed
        print("\n🧠 Generating embeddings...")
        embedder = Embedder()
        chunks_with_embeddings = embedder.embed_chunks(chunks)
        
        # Store
        print("\n💾 Storing in vector database...")
        store = VectorStore()
        await store.connect()
        await store.insert_documents(chunks_with_embeddings)
        await store.close()
        
        print("\n✅ All data processed and stored successfully!")

        if __name__ == "__main__":
            import asyncio
            asyncio.run(process_all_data())
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            print("✅ Database connection closed")

