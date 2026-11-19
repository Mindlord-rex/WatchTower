import logging
import zipfile
import threading
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import clip
from ultralytics import YOLO

from src import config

logger = logging.getLogger("watchtower.models")


class ModelManager:
    """
    Thread-safe Singleton class managing YOLOv8 person detection and CLIP embedding models.
    Supports asynchronous background warmup and lazy loading on demand without blocking
    the FastAPI server startup.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._is_loading = False
                    cls._instance._load_event = threading.Event()
                    cls._instance.device = config.DEVICE
                    cls._instance.yolo = None
                    cls._instance.clip_model = None
                    cls._instance.clip_preprocess = None
        return cls._instance

    def load_models(self):
        """Loads and warms up YOLOv8 and CLIP models."""
        with self._lock:
            if self._initialized:
                return
            if self._is_loading:
                self._load_event.wait()
                return
            self._is_loading = True

        try:
            logger.info(f"Initializing ModelManager on device: {self.device}")

            # Ensure YOLO weights are extracted
            self._ensure_yolo_weights()

            # Load YOLO model
            try:
                logger.info(f"Loading YOLO model from {config.YOLO_WEIGHTS_PATH}...")
                self.yolo = YOLO(str(config.YOLO_WEIGHTS_PATH))
                logger.info(f"YOLO model loaded successfully. Classes: {self.yolo.names}")
            except Exception as e:
                logger.error(f"Failed to load custom YOLO weights ({e}). Falling back to 'yolov8n.pt'")
                self.yolo = YOLO("yolov8n.pt")

            # Load CLIP model
            try:
                local_clip_file = config.CLIP_CACHE_DIR / "ViT-B-32.pt"
                if local_clip_file.is_file():
                    logger.info(f"Loading local CLIP model checkpoint from {local_clip_file}...")
                    clip_target = str(local_clip_file)
                else:
                    logger.info(f"Loading CLIP ViT-B/32 model (download_root: {config.CLIP_CACHE_DIR})...")
                    clip_target = "ViT-B/32"

                self.clip_model, self.clip_preprocess = clip.load(
                    clip_target,
                    device=self.device,
                    jit=False,
                    download_root=str(config.CLIP_CACHE_DIR)
                )
                self.clip_model.eval()
                logger.info("CLIP model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load CLIP ViT-B/32 model: {e}")
                raise e

            self._initialized = True
            logger.info("All WatchTower models successfully loaded and warmed up.")
        finally:
            self._is_loading = False
            self._load_event.set()

    def ensure_loaded(self):
        """Ensures models are loaded before inference."""
        if not self._initialized:
            self.load_models()

    def is_ready(self) -> bool:
        """Returns True if models are ready for inference."""
        return self._initialized

    def _ensure_yolo_weights(self):
        """Extract best.pt from run_yolo_checkpoint.zip if it does not already exist."""
        if not config.YOLO_WEIGHTS_PATH.exists():
            if config.YOLO_ZIP_PATH.exists():
                logger.info(f"Extracting PRW detector weights from {config.YOLO_ZIP_PATH}...")
                config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(config.YOLO_ZIP_PATH, 'r') as z:
                    target_member = "detect/prw_detector/yolov8n_prw3/weights/best.pt"
                    if target_member in z.namelist():
                        with z.open(target_member) as src, open(config.YOLO_WEIGHTS_PATH, "wb") as dst:
                            dst.write(src.read())
                        logger.info(f"Extracted {config.YOLO_WEIGHTS_PATH}")
                    else:
                        for name in z.namelist():
                            if name.endswith("best.pt"):
                                with z.open(name) as src, open(config.YOLO_WEIGHTS_PATH, "wb") as dst:
                                    dst.write(src.read())
                                logger.info(f"Extracted alternative {name} -> {config.YOLO_WEIGHTS_PATH}")
                                break
            else:
                logger.warning(f"Zip archive {config.YOLO_ZIP_PATH} not found.")

    def detect_persons(self, frame_rgb: np.ndarray, conf_thresh: float = config.DEFAULT_CONF_THRESHOLD) -> List[Tuple[int, int, int, int, float]]:
        """
        Runs YOLO person detector on RGB image frame.
        Returns list of (x1, y1, x2, y2, confidence).
        Filters only person class (class_id 0 in PRW detector or standard YOLO person class).
        """
        self.ensure_loaded()
        results = self.yolo(frame_rgb, verbose=False)[0]
        detections = []
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf < conf_thresh:
                continue

            cls_id = int(box.cls.item())
            # PRW detector has only 1 class (0: person). Standard YOLO has 0: person.
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            h, w = frame_rgb.shape[:2]
            # Clamp coordinates to frame boundary
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 > x1 and y2 > y1:
                detections.append((x1, y1, x2, y2, conf))

        return detections

    def encode_crops_batch(self, crops_pil: List[Image.Image]) -> List[List[float]]:
        """
        Generates normalized CLIP embeddings for a batch of PIL Image crops.
        """
        if not crops_pil:
            return []

        self.ensure_loaded()
        tensors = [self.clip_preprocess(crop) for crop in crops_pil]
        batch_tensor = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            embeddings = self.clip_model.encode_image(batch_tensor)
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            emb_list = embeddings.cpu().numpy().tolist()

        return emb_list

    def encode_text_query(self, query_text: str) -> List[float]:
        """
        Generates normalized CLIP text embedding for a natural-language search query.
        """
        self.ensure_loaded()
        with torch.no_grad():
            text_tokens = clip.tokenize([query_text], truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            # Normalize embedding for cosine similarity
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().flatten().tolist()


# Global accessor
def get_model_manager() -> ModelManager:
    return ModelManager()
