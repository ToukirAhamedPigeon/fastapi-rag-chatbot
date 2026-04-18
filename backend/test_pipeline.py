import sys
import os

# Add path
sys.path.append('D:/fastapi-rag-chatbot/backend')

# Test imports
print("1. Testing imports...")
try:
    from app.services.data_loader import DataLoader
    print("   ✅ DataLoader imported")
except Exception as e:
    print(f"   ❌ DataLoader failed: {e}")

try:
    from app.services.chunker import TextChunker
    print("   ✅ TextChunker imported")
except Exception as e:
    print(f"   ❌ TextChunker failed: {e}")

try:
    from app.services.embedder import Embedder
    print("   ✅ Embedder imported")
except Exception as e:
    print(f"   ❌ Embedder failed: {e}")

try:
    from app.services.vector_store import VectorStore
    print("   ✅ VectorStore imported")
except Exception as e:
    print(f"   ❌ VectorStore failed: {e}")

# Test data loading
print("\n2. Testing data loading...")
try:
    loader = DataLoader("D:/fastapi-rag-chatbot/data/raw/")
    docs = loader.load_all()
    print(f"   ✅ Loaded {len(docs)} documents")
    if docs:
        print(f"   📝 Sample title: {docs[0].get('title', 'N/A')}")
except Exception as e:
    print(f"   ❌ Data loading failed: {e}")

print("\n✅ Test complete!")