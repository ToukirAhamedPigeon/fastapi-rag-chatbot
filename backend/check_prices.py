import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# Check price range
cur.execute('SELECT MIN((metadata->>%s)::float), MAX((metadata->>%s)::float) FROM documents', ('price', 'price'))
min_price, max_price = cur.fetchone()
print(f'Price range: ${min_price} - ${max_price}')

# Check products under $100
cur.execute("SELECT title, (metadata->>'price')::float as price FROM documents WHERE (metadata->>'price')::float < 100 LIMIT 10")
print('\nProducts under $100:')
for row in cur.fetchall():
    print(f'  - {row[0]}: ${row[1]}')

# Count products under $50
cur.execute("SELECT COUNT(*) FROM documents WHERE (metadata->>'price')::float < 50")
count_under_50 = cur.fetchone()[0]
print(f'\nTotal products under $50: {count_under_50}')

cur.close()
conn.close()