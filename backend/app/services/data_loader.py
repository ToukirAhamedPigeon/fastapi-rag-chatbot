import pandas as pd
import os
from typing import List, Dict, Any
from PyPDF2 import PdfReader
import chardet

class DataLoader:
    def __init__(self, data_path: str = "D:/fastapi-rag-chatbot/data/raw/"):
       self.data_path = data_path
    
    def load_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Load CSV file and convert to documents"""
        df = pd.read_csv(file_path)
        documents = []
        for _, row in df.iterrows():
            doc = {
                "id": str(row.get('id', len(documents))),
                "title": row.get('title', 'Untitled'),
                "content": self._create_content(row),
                "category": row.get('category', 'General'),
                "price": row.get('price', 0),
                "rating": row.get('rating', 0),
                "brand": row.get('brand', ''),
                "tags": row.get('tags', '')
            }
            documents.append(doc)
        return documents
    
    def _create_content(self, row: pd.Series) -> str:
        """Create searchable content from row data"""
        content_parts = [
            f"Title: {row.get('title', '')}",
            f"Category: {row.get('category', '')}",
            f"Description: {row.get('description', '')}",
            f"Price: ${row.get('price', 0)}",
            f"Brand: {row.get('brand', '')}",
            f"Features: {row.get('features', '')}"
        ]
        return "\n".join(content_parts)
    
    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Load PDF file and extract text"""
        reader = PdfReader(file_path)
        documents = []
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        # Split PDF into chunks
        chunks = self._chunk_text(text, chunk_size=1000, overlap=150)
        for i, chunk in enumerate(chunks):
            doc = {
                "id": f"{os.path.basename(file_path)}_{i}",
                "title": os.path.basename(file_path),
                "content": chunk,
                "category": "PDF Document",
                "source": file_path
            }
            documents.append(doc)
        return documents
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def load_all(self) -> List[Dict[str, Any]]:
        """Load all documents from data folder"""
        all_documents = []
        
        # Load CSV
        csv_path = os.path.join(self.data_path, "ecommerce_products.csv")
        if os.path.exists(csv_path):
            print(f"📄 Loading CSV: {csv_path}")
            docs = self.load_csv(csv_path)
            all_documents.extend(docs)
            print(f"   ✅ Loaded {len(docs)} documents")
        
        # Load PDFs (if any)
        pdf_files = [f for f in os.listdir(self.data_path) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.data_path, pdf_file)
            print(f"📄 Loading PDF: {pdf_file}")
            docs = self.load_pdf(pdf_path)
            all_documents.extend(docs)
            print(f"   ✅ Loaded {len(docs)} chunks")
        
        return all_documents

# Test
if __name__ == "__main__":
    loader = DataLoader("data/raw/")
    docs = loader.load_all()
    print(f"\n📊 Total documents loaded: {len(docs)}")
    print(f"📝 Sample document:")
    print(docs[0] if docs else "No documents found")