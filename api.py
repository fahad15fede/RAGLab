import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from viper_graph import graph

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="VIPERBot API", lifespan=lifespan)
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=128)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = graph.invoke({"messages": [HumanMessage(content=payload.message)]}, config={"configurable": {"thread_id": payload.session_id}})
        return {"answer": result["answer"], "sources": result.get("sources", []), "session_id": payload.session_id}
    except Exception as exc:
        print(f"Chat failure: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail="Chat service temporarily unavailable") from None
