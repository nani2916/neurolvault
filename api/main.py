from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import predictions, chatbot, orchestrator_api
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="NeuralVault AI Platform",
    description="AI-powered loan risk intelligence platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    predictions.router,
    prefix="/api/predictions",
    tags=["predictions"]
)

app.include_router(
    chatbot.router,
    prefix="/api/chatbot",
    tags=["chatbot"]
)

app.include_router(
    orchestrator_api.router,
    prefix="/api/orchestrator",
    tags=["orchestrator"]
)

@app.get("/")
async def root():
    return {
        "message": "NeuralVault AI Platform",
        "version": "1.0.0",
        "status": "running"
    }
