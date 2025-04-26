from fastapi import FastAPI, Request, Form, APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os, csv
from datetime import datetime
from service_for_main import *

# Define lifespan handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models and boards
    app.state.sorted_boards = get_sorted_boards()
    app.state.embedding_model = get_embedding_model()
    app.state.classify_model = get_classify_model(
        app.state.embedding_model,
        app.state.sorted_boards
    )
    yield
    # Shutdown: nothing to clean up


# Create FastAPI app with custom lifespan
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "boards": request.app.state.sorted_boards}
    )


api = APIRouter(prefix="/api/model")

@api.get("/prediction")
async def api_prediction(title: str):
    cleaned = clean_title(title)
    tokens  = tokenize(cleaned)
    embedded = embed(app.state.embedding_model, tokens)
    pred = predict(
        app.state.classify_model,
        app.state.sorted_boards,
        embedded
    )
    return JSONResponse({"prediction": pred})

@api.post("/feedback")
async def feedback_view(payload: dict):
    title = payload.get("title", "")
    pred  = payload.get("predicted_board", "")
    actual= payload.get("actual_board", "")
    # Append feedback to CSV
    feedback_file = "./feedback/user-labeled-titles.csv"
    first = not os.path.exists(feedback_file)
    with open(feedback_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if first:
            writer.writerow(["timestamp","title","predicted_board","actual_board"])
        writer.writerow([
            datetime.now().isoformat(),
            title, 
            pred, actual
        ])
    return JSONResponse({"status": "ok"})

@api.get("/boards")
async def api_boards():
    return JSONResponse({"boards": app.state.sorted_boards})

app.include_router(api)