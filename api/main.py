"""
FastAPI Application Entry Point.

Provides REST API server for sales forecasting inference and metadata querying.
Loads trained model artifact into memory once during application startup.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from src.ingestion.logger import setup_logger
from src.forecasting.model_utils import load_model_artifact

logger = setup_logger("api_logger")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager.
    Loads trained model artifact once on startup into memory.
    """
    logger.info("Initializing Sales Forecasting REST API...")
    try:
        artifact = load_model_artifact()
        app.state.model_artifact = artifact
        logger.info(f"Successfully loaded model artifact into memory (Version {artifact.get('model_version', '1.0.0')}).")
    except FileNotFoundError:
        logger.warning("Trained model artifact not found during startup. API will attempt dynamic loading on request.")
        app.state.model_artifact = None
    except Exception as e:
        logger.error(f"Error loading model artifact during startup: {str(e)}", exc_info=True)
        app.state.model_artifact = None
        
    yield
    
    logger.info("Shutting down Sales Forecasting API server.")

app = FastAPI(
    title="Sales Forecasting API",
    description="Production-grade REST API serving machine learning sales forecasts and model information.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)

@app.get("/", tags=["root"])
def read_root():
    """
    Root endpoint returning service status and documentation links.
    """
    return {
        "message": "Sales Forecasting REST API is online.",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_url": "/health",
        "model_info_url": "/model-info"
    }
