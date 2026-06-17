import sys
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings("ignore")
# Ensure parent directory is in the path to run backend directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from game_ops.api.database import init_db
from game_ops.api.seed_data import seed_database
from game_ops.main_pipeline import GameOpsPredictor
from game_ops.api.routes import scores, leaderboard, players, analysis

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler to initialize database schema, load ML models,
    and seed PostgreSQL on startup.
    """
    # 1. Startup: Initialize DB tables
    init_db()
    
    # 2. Load ML Models
    predictor = GameOpsPredictor()
    predictor.load_models()
    app.state.predictor = predictor
    print("--------------------------------------------------")
    print("Machine Learning models loaded successfully.")
    
    # 3. Auto-seed the database with 10,000 processed players
    seed_database()
    print("Database auto-seeded with 10,000 processed players.")
    print("Docs available at http://localhost:8000/docs")
    print("--------------------------------------------------")
    yield
    # Shutdown (no actions required)

app = FastAPI(
    title="Game Operations Intelligence System API",
    description="REST backend for real-time score submissions, cheater detection, and matchmaking.",
    version="1.0.0",
    lifespan=lifespan
)

# Custom RequestValidationError handler to return HTTP 400 Bad Request
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Invalid input provided",
            "errors": exc.errors()
        }
    )

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(scores.router, tags=["Scores"])
app.include_router(leaderboard.router, tags=["Leaderboard"])
app.include_router(players.router, tags=["Players"])
app.include_router(analysis.router, tags=["Analysis"])

if __name__ == "__main__":
    import uvicorn
    # Local run logic
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
