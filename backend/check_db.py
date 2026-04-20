import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

try:
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    
    cur.execute('SELECT COUNT(*) FROM documents')
    count = cur.fetchone()[0]
    print(f'Total documents: {count}')
    
    if count > 0:
        cur.execute("SELECT title, metadata->>'category' FROM documents LIMIT 3")
        print('\nSample documents:')
        for row in cur.fetchall():
            print(f'  - {row[0]} ({row[1]})')
    
    cur.close()
    conn.close()
    
except Exception as e:
    print(f'Error: {e}')