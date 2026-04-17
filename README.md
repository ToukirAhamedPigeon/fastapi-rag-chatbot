# FastAPI RAG Chatbot

AI-powered chatbot using RAG (Retrieval-Augmented Generation) with 1000+ e-commerce articles.

## Tech Stack

### Backend
- FastAPI (Python)
- Groq Cloud (LLM)
- Neon PostgreSQL + pgvector
- Upstash Redis
- Firebase Auth

### Frontend
- React 18
- Tailwind CSS
- Firebase SDK
- Axios

## Project Structure
backend/ # FastAPI backend
frontend/ # React frontend
data/ # Datasets and embeddings
scripts/ # Setup and deployment scripts


## Local Development

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload

### Frontend
cd frontend
npm install
npm run dev

### Deployment
Backend: Render.com

Frontend: Vercel

Database: Neon.tech

Cache: Upstash Redis

### Environment Variables
Copy .env.example to .env and fill:

GROQ_API_KEY

DATABASE_URL (Neon)

REDIS_URL (Upstash)

FIREBASE_CONFIG (JSON)

### License
MIT