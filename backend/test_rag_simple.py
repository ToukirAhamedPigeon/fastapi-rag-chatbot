import sys
sys.path.append('D:/fastapi-rag-chatbot/backend')
from app.services.rag_engine_simple import RAGEngineSimple

rag = RAGEngineSimple()

queries = [
    'What products are under $50?',
    'Tell me about books',
    'What is the price of Jeans?'
]

for q in queries:
    print('=' * 50)
    print('Q:', q)
    try:
        result = rag.ask(q)
        print('A:', result['answer'][:300])
        print('Sources:', result['sources'])
    except Exception as e:
        print(f'Error: {e}')
    print()