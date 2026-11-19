import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage directories
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
FRAMES_DIR = BASE_DIR / "storage" / "frames"
CROPS_DIR = BASE_DIR / "storage" / "crops"
CHROMA_DIR = BASE_DIR / "chroma_db"
DATABASE_PATH = BASE_DIR / "video.db"

# Model paths and cache directories
MODEL_DIR = BASE_DIR / "model"
WEIGHTS_DIR = MODEL_DIR / "weights"
YOLO_WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
YOLO_ZIP_PATH = MODEL_DIR / "run_yolo_checkpoint.zip"
CLIP_CACHE_DIR = MODEL_DIR / "clip_cache"
TORCH_CACHE_DIR = MODEL_DIR / "torch_cache"
ULTRALYTICS_DIR = BASE_DIR / "Ultralytics"

# Environment configuration
os.environ["YOLO_CONFIG_DIR"] = str(ULTRALYTICS_DIR)
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_VERBOSE"] = "false"
os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)

# Ensure directories exist
for p in [UPLOAD_DIR, FRAMES_DIR, CROPS_DIR, CHROMA_DIR, WEIGHTS_DIR, CLIP_CACHE_DIR, TORCH_CACHE_DIR, ULTRALYTICS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Hardware device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Processing defaults
DEFAULT_FRAME_INTERVAL = 30  # Sample every 30th frame (~1 fps for 30fps video)
DEFAULT_CONF_THRESHOLD = 0.40  # YOLO detection confidence threshold
DEFAULT_EMBEDDING_BATCH_SIZE = 64  # Batch size for CLIP inference
CHROMA_COLLECTION_NAME = "watchtower_persons"
