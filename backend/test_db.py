import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def test_connection():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        result = await conn.fetch('SELECT version()')
        print("✅ Database connected successfully!")
        print(f"   PostgreSQL version: {result[0]['version']}")
        
        # Check pgvector
        vector_check = await conn.fetch('SELECT extname FROM pg_extension WHERE extname = \'vector\'')
        if vector_check:
            print("✅ pgvector extension enabled")
        else:
            print("⚠️ pgvector extension not found")
        
        await conn.close()
    except Exception as e:
        print(f"❌ Connection failed: {e}")

asyncio.run(test_connection())