# WatchTower – Video Intelligence & Semantic Person Search

WatchTower is a web application designed for surveillance footage analysis. It allows users to upload surveillance recordings, automatically detect and index people using computer vision models, and search for individuals across all indexed videos using natural-language descriptions.

---

## Key Features

- **Flexible Video Ingestion:**
  - **Local Video Upload:** Drag-and-drop or select MP4, MOV, and MKV files.
  - **YouTube & URL Import:** Download and index directly from YouTube URLs (`https://www.youtube.com/watch?v=...`) or remote video streams powered by `yt-dlp`.
- **Semantic Person Search:** Search across all indexed videos using descriptive natural language (e.g., *"person wearing a red jacket"*, *"man carrying a black backpack"*, *"someone walking with a suitcase"*).
- **Computer Vision Models:**
  - **YOLOv8n (PRW):** Fine-tuned specifically for person re-identification in surveillance footage (single-class person detector).
  - **OpenAI CLIP (ViT-B/32):** Multimodal text and image embeddings in a shared 512-dimensional vector space.
- **Persistent Vector & Metadata Storage:**
  - **ChromaDB:** Persistent vector database storing normalized CLIP embeddings with HNSW cosine similarity space.
  - **SQLite:** Persistent relational metadata storage for video specifications, processing states, frame indices, and bounding box coordinates.
- **JavaScript-Free Frontend:**
  - Powered by **FastAPI**, **Jinja2** templates, **Tailwind CSS**, and **HTMX**.
  - **Zero custom JavaScript** and no third-party JavaScript UI frameworks.
  - Native HTML5 `<video>` player with browser URL fragments (`#t=timestamp`) and HTTP 206 Partial Content range streaming for precise timestamp seeking.
- **Asynchronous Processing:**
  - Non-blocking frame extraction, person cropping, and batch embedding generation.
  - Real-time progress updates via automated HTMX polling (`hx-trigger="every 2s"`).
- **Cascading Deletion:**
  - Deleting a video cleans up its SQLite records, ChromaDB vector embeddings, stored original video, extracted crops, and frame preview images.

---

## Architecture Overview

```
                      +-----------------------------+
                      |     Client Browser (HTMX)    |
                      |  (Completely JavaScript-Free)|
                      +--------------+--------------+
                                     |  HTTP / HTMX
                                     v
                      +-----------------------------+
                      |        FastAPI Server       |
                      |  Routes, Streaming & Jinja2 |
                      +----+-------------------+----+
                           |                   |
            +--------------+                   +---------------+
            |                                                  |
            v                                                  v
+-------------------------+                       +-------------------------+
|     Video Processor     |                       |      Search Engine      |
| - OpenCV frame sampling |                       | - CLIP text tokenizer   |
| - YOLOv8 Person Detector|                       | - Cosine similarity     |
| - CLIP Crop Embeddings  |                       | - Ranked metadata join  |
+------------+------------+                       +------------+------------+
             |                                                 |
             +-----------------------+-------------------------+
                                     |
                                     v
                   +-----------------------------------+
                   |          Storage Layer            |
                   | - ChromaDB (Vector Embeddings)    |
                   | - SQLite (video.db Metadata)      |
                   | - File System (storage/ & uploads)|
                   +-----------------------------------+
```

---

## Directory Structure

```
WatchTower/
├── model/
│   ├── run_yolo_checkpoint.zip  # Trained YOLOv8 checkpoint archive
│   ├── weights/
│   │   └── best.pt              # Extracted YOLOv8 PRW detector weights
│   └── clip_cache/
│       └── ViT-B-32.pt          # Local CLIP ViT-B/32 model weights
├── src/
│   ├── config.py                # Global configurations, directories, device setup
│   ├── database.py              # SQLite schema, queries, cascade deletion
│   ├── models.py                # Thread-safe ModelManager (YOLOv8 + CLIP)
│   ├── vector_db.py             # ChromaDB vector collection and queries
│   ├── video_processor.py       # OpenCV decoder, crop extraction, batch embedder
│   ├── search.py                # CLIP semantic query encoder & vector retrieval
│   ├── streaming.py             # HTTP 206 Range video streaming handler
│   └── main.py                  # FastAPI application & web endpoints
├── templates/
│   ├── base.html                # Master layout with Tailwind & HTMX
│   ├── dashboard.html           # Surveillance dashboard & stats overview
│   ├── upload.html              # HTML5 drag-and-drop video upload form
│   ├── search.html              # Natural-language semantic search interface
│   ├── library.html             # Video library listing & management
│   ├── player.html              # HTML5 video player with timestamp navigation
│   └── partials/
│       ├── video_status.html    # HTMX polling card component
│       └── search_results.html  # Dynamic search results grid
├── static/
│   └── css/
│       └── styles.css           # Custom dark theme styles
├── storage/
│   ├── frames/                  # Annotated preview frames with bounding boxes
│   └── crops/                   # Cropped person images
├── uploads/                     # Uploaded surveillance video files
├── chroma_db/                   # Persistent ChromaDB vector database
├── video.db                     # SQLite relational database
├── requirements.txt             # Project dependencies
└── README.md                    # Documentation
```

---

## Prerequisites & Installation

### 1. Requirements
- Python 3.10 or 3.11
- CUDA-compatible GPU (recommended for faster batch inference, CPU fallback supported)
- FFmpeg / OpenCV video codec support

### 2. Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/Mindlord-rex/WatchTower.git
cd WatchTower

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Model Weights Setup

The application automatically verifies and initializes all required weights on first launch:

1. **YOLOv8 (PRW Detector):**
   - The custom model weights are packaged inside `model/run_yolo_checkpoint.zip`.
   - On initial run, `src/models.py` automatically extracts `detect/prw_detector/yolov8n_prw3/weights/best.pt` to `model/weights/best.pt`.
2. **CLIP (ViT-B/32):**
   - Pre-downloaded model weights are stored locally in `model/clip_cache/ViT-B-32.pt`.
   - The application loads this local checkpoint directly in offline mode (`jit=False`), avoiding any runtime network dependencies.

---

## Running the Application

Start the local FastAPI development server using Uvicorn:

```bash
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

Once started, open your web browser and navigate to:
```
http://127.0.0.1:8000
```

---

## How It Works

### 1. Video Ingestion & Processing
1. **Upload or Remote Import:**
   - **Local File:** Uploaded directly through the browser interface.
   - **YouTube / URL:** Ingested asynchronously from remote links using `yt-dlp`.
   - The file is saved under `uploads/` with a unique UUID.
2. An asynchronous background task launches `process_video_pipeline`.
3. OpenCV reads frames sequentially according to the configurable `sampling_interval` (default: every 30 frames, ~1 frame per second).
4. YOLOv8 detects person instances with scores above `conf_thresh` (default: 0.40).
5. For each detection:
   - The person is cropped and saved to `storage/crops/{video_id}/`.
   - An annotated frame with highlighted bounding boxes is saved to `storage/frames/{video_id}/`.
   - CLIP computes normalized 512-D embeddings in batches.
   - Vector embeddings are indexed in ChromaDB (`watchtower_persons` collection).
   - Timestamp and bounding box coordinates are persisted in SQLite.
6. The frontend displays processing progress in real-time using HTMX polling (`hx-trigger="every 2s"`).

### 2. Semantic Person Search
1. The user enters a natural-language query (e.g., *"person wearing a red jacket"*).
2. The CLIP text tokenizer and text encoder transform the query into a 512-D vector.
3. ChromaDB executes a cosine distance search across all indexed person embeddings (or filtered by a specific video).
4. Detections are ranked by cosine similarity ($1 - \text{distance}$) and rendered dynamically into the search grid via HTMX.

### 3. Timestamp Navigation (Pure HTML5)
- When clicking any search result or detection thumbnail, the user is navigated to `/videos/{video_id}/player?t={timestamp}`.
- The server renders an HTML5 `<video>` tag with the target timestamp set via standard browser URL fragment:
  ```html
  <video src="/videos/{video_id}/stream#t={timestamp}" controls autoplay>
  ```
- The backend `/videos/{video_id}/stream` endpoint implements HTTP 206 Partial Content range requests, allowing the browser to seek directly to the exact second in the surveillance recording without downloading the entire video first.

---

## Known Limitations & Tips

- **Video Codecs:** Browser video playback requires browser-compatible codecs (H.264 / AAC for MP4). If a video uses an exotic surveillance DVR codec, transcode it to H.264 MP4 prior to uploading.
- **CPU Inference Speed:** On systems without an NVIDIA GPU, processing runs on the CPU. Setting a larger sampling interval (e.g., 60 or 90 frames) significantly increases processing speed for long recordings.
- **Offline Mode:** The application is configured to run fully offline with zero remote telemetry or external API calls.
