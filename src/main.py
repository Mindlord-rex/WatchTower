import os
import uuid
import shutil
import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    HTTPException,
    status
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src import config
from src.database import (
    init_db,
    get_dashboard_stats,
    list_videos,
    get_video,
    create_video_record,
    delete_video_record,
    get_db_connection
)
from src.video_processor import process_video_pipeline, delete_video_assets
from src.search import semantic_person_search, format_timestamp
from src.streaming import stream_video_file
from src.models import get_model_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("watchtower.app")

# Initialize FastAPI app
app = FastAPI(
    title="WatchTower",
    description="Video Intelligence & Semantic Person Search Platform",
    version="1.0.0"
)

# Mount static and storage asset directories
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "static")), name="static")
app.mount("/storage", StaticFiles(directory=str(config.BASE_DIR / "storage")), name="storage")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=str(config.BASE_DIR / "templates"))


@app.on_event("startup")
def on_startup():
    """Fast non-blocking startup: initialize DB and warm models in background daemon thread."""
    logger.info("Initializing WatchTower database...")
    init_db()

    def _background_warmup():
        try:
            logger.info("Background model warmup started...")
            get_model_manager().load_models()
            logger.info("Background model warmup completed successfully.")
        except Exception as e:
            logger.error(f"Background model warmup failed: {e}")

    threading.Thread(target=_background_warmup, daemon=True).start()
    logger.info("FastAPI web server startup complete (models warming in background).")


# ============================================================================
# Page Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Dashboard page showing system overview, stats, and recently uploaded videos."""
    stats = get_dashboard_stats()
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "active_page": "dashboard",
            "stats": stats
        }
    )


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Video upload page."""
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "active_page": "upload"
        }
    )


@app.post("/upload")
async def handle_video_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sampling_interval: int = Form(config.DEFAULT_FRAME_INTERVAL),
    conf_thresh: float = Form(config.DEFAULT_CONF_THRESHOLD)
):
    """
    Handles surveillance video upload, saves file, registers database entry,
    and dispatches asynchronous background processing task.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    # Generate unique video ID and safe storage path
    video_id = str(uuid.uuid4())
    safe_filename = Path(file.filename).name
    extension = Path(safe_filename).suffix.lower()
    allowed_extensions = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{extension}'. Please upload MP4, MOV, or MKV."
        )

    saved_filename = f"{video_id}_{safe_filename}"
    file_path = config.UPLOAD_DIR / saved_filename

    # Save uploaded file in chunks
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    finally:
        file.file.close()

    # Create SQLite record
    create_video_record(
        video_id=video_id,
        filename=safe_filename,
        file_path=str(file_path),
        sampling_interval=sampling_interval
    )

    # Dispatch non-blocking background processing pipeline
    background_tasks.add_task(
        process_video_pipeline,
        video_id=video_id,
        sampling_interval=sampling_interval,
        conf_thresh=conf_thresh
    )

    # Redirect directly to video library to view live progress
    return RedirectResponse(url="/library", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/search", response_class=HTMLResponse)
async def search_page(
    request: Request,
    q: Optional[str] = None,
    video_id: Optional[str] = None,
    top_k: int = 12
):
    """Semantic search page. Can optionally execute initial search if 'q' is provided."""
    videos = list_videos()
    initial_results = None
    if q and q.strip():
        initial_results = semantic_person_search(
            query_text=q.strip(),
            top_k=top_k,
            video_id=video_id if video_id else None
        )

    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "active_page": "search",
            "videos": videos,
            "selected_video_id": video_id or "",
            "initial_query": q or "",
            "initial_results": initial_results
        }
    )


@app.post("/search", response_class=HTMLResponse)
async def handle_search_htmx(
    request: Request,
    query: str = Form(...),
    top_k: int = Form(12),
    video_id: Optional[str] = Form(None)
):
    """
    HTMX endpoint for semantic search.
    Returns rendered HTML partial containing results grid.
    """
    target_video = video_id if (video_id and video_id.strip()) else None
    results = semantic_person_search(
        query_text=query,
        top_k=top_k,
        video_id=target_video
    )

    return templates.TemplateResponse(
        "partials/search_results.html",
        {
            "request": request,
            "query": query,
            "results": results
        }
    )


@app.get("/library", response_class=HTMLResponse)
async def library_page(request: Request):
    """Video library page displaying all uploaded videos and statuses."""
    videos = list_videos()
    return templates.TemplateResponse(
        "library.html",
        {
            "request": request,
            "active_page": "library",
            "videos": videos
        }
    )


@app.get("/videos/{video_id}/status", response_class=HTMLResponse)
async def video_status_htmx(request: Request, video_id: str):
    """
    HTMX polling endpoint. Returns updated single video card HTML partial.
    Stops polling automatically when processing completes or fails.
    """
    video = get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return templates.TemplateResponse(
        "partials/video_status.html",
        {
            "request": request,
            "video": video
        }
    )


@app.get("/videos/{video_id}/player", response_class=HTMLResponse)
async def video_player_page(
    request: Request,
    video_id: str,
    t: float = 0.0
):
    """
    Native HTML5 video playback page.
    Renders player with URL fragment `#t={t}` and list of detected persons.
    """
    video = get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Fetch detections for this video
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM detections
        WHERE video_id = ?
        ORDER BY frame_number ASC
    """, (video_id,))
    rows = c.fetchall()
    conn.close()

    detections = []
    for r in rows:
        d = dict(r)
        d["timestamp_formatted"] = format_timestamp(d["timestamp"])
        detections.append(d)

    return templates.TemplateResponse(
        "player.html",
        {
            "request": request,
            "active_page": "library",
            "video": video,
            "detections": detections,
            "current_timestamp": t,
            "formatted_timestamp": format_timestamp(t)
        }
    )


@app.get("/videos/{video_id}/stream")
async def stream_video(request: Request, video_id: str):
    """
    Streams original video file with HTTP 206 Partial Content range support.
    Enables native HTML5 seeking and timestamp navigation.
    """
    video = get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    file_path = video["file_path"]
    return stream_video_file(file_path, request)


@app.post("/videos/{video_id}/delete")
async def delete_video(video_id: str):
    """
    Cascades video deletion:
    - Deletes SQLite detection & video records.
    - Deletes ChromaDB vector embeddings.
    - Deletes saved original video file, crops, and annotated frames.
    """
    video = delete_video_record(video_id)
    if video:
        # Delete original file
        orig_file = Path(video["file_path"])
        if orig_file.exists():
            try:
                os.remove(orig_file)
            except Exception as e:
                logger.warning(f"Failed to remove file {orig_file}: {e}")

        # Delete frames, crops, and vector DB entries
        delete_video_assets(video_id)

    return RedirectResponse(url="/library", status_code=status.HTTP_303_SEE_OTHER)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=False)
