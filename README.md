# VIPERBot — FastAPI + LangGraph + Chroma Cloud

This project adapts the existing VIPER LangGraph RAG chatbot to a hosted API.

## Setup
1. Create a virtual environment and run `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and enter your private credentials. Do not commit `.env`.
3. Ensure your Chroma Cloud database has collection `viper_documentation` indexed with **gemini-embedding-001, 768 dimensions**. If empty, run `python vector_store.py` once (Gemini quota applies).
4. Start with `uvicorn api:app --reload --port 8000`.
5. Test `GET /health` and POST `/chat` with JSON `{"message":"What is fang?","session_id":"test-session"}`.

## Render
Create a Python Web Service from the GitHub repo. Build: `pip install -r requirements.txt`; Start: `uvicorn api:app --host 0.0.0.0 --port $PORT`. Add environment variables in Render. No local database disk is needed for vectors.

## Caveats
- LangGraph `InMemorySaver` keeps conversation history only until process restart; use durable checkpoint storage if persistence is required.
- The existing hybrid retrieval scans Chroma documents for exact keyword matches; this may be slow for large collections.
- Protect the public API with rate limits/authentication before broad public launch.
- The Gemini model default is configurable; select a model available to your API key.
- Ensure Chroma collection has compatible 768-dimensional embeddings.
