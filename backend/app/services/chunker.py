from typing import List, Dict, Any
import re

class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into smaller chunks"""
        chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            
            # Split by paragraphs first
            paragraphs = content.split('\n\n')
            
            current_chunk = ""
            current_metadata = {
                'source_id': doc.get('id'),
                'title': doc.get('title'),
                'category': doc.get('category', 'General')
            }
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < self.chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append({
                            'content': current_chunk.strip(),
                            'metadata': current_metadata.copy()
                        })
                    
                    # Start new chunk with overlap
                    words = current_chunk.split()
                    overlap_text = " ".join(words[-self.overlap:]) if len(words) > self.overlap else current_chunk
                    current_chunk = overlap_text + "\n\n" + para + "\n\n"
            
            # Add last chunk
            if current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': current_metadata.copy()
                })
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        lengths = [len(chunk['content']) for chunk in chunks]
        return {
            'total_chunks': len(chunks),
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0
        }

# Test
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load documents
    loader = DataLoader("data/raw/")
    docs = loader.load_all()
    
    # Chunk them
    chunker = TextChunker(chunk_size=1000, overlap=150)
    chunks = chunker.chunk_documents(docs)
    
    print(f"📊 Chunking Statistics:")
    stats = chunker.get_chunk_stats(chunks)
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n📝 Sample chunk:")
    print(chunks[0]['content'][:500] + "...")
    print(f"   Metadata: {chunks[0]['metadata']}")