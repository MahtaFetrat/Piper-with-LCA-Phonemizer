import uvicorn

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from contextlib import asynccontextmanager

from config import TTS_CONFIG, FASTAPI_CONFIG, TAGS_METADATA, SERVER_CONFIG
from core import load_voices_config, clear_model_cache
from routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting TTS Microservice...")
    TTS_CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)
    load_voices_config()
    print("✅ TTS Microservice ready!")
    yield
    print("🔄 Shutting down TTS Microservice...")
    clear_model_cache()
    print("✅ TTS Microservice shutdown complete!")

app = FastAPI(
    title=FASTAPI_CONFIG["title"],
    version=FASTAPI_CONFIG["version"],
    lifespan=lifespan,
    openapi_tags=TAGS_METADATA,
    contact=FASTAPI_CONFIG["contact"],
    license_info=FASTAPI_CONFIG["license_info"],
    openapi_url="/api/tts/openapi.json",
    docs_url="/api/tts/docs",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        reload=SERVER_CONFIG["reload"]
    )
