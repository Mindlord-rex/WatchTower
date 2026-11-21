import cv2
import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image

from src import config
from src.models import get_model_manager
from src.database import (
    update_video_progress,
    update_video_metadata,
    save_detections_batch,
    get_video
)
from src.vector_db import get_vector_db

logger = logging.getLogger("watchtower.processor")


def process_video_pipeline(
    video_id: str,
    sampling_interval: int = config.DEFAULT_FRAME_INTERVAL,
    conf_thresh: float = config.DEFAULT_CONF_THRESHOLD
):
    """
    Synchronous processing function intended to run in a background worker thread.
    Reads frames with OpenCV, runs YOLO person detector, crops persons,
    computes CLIP embeddings in batches, and indexes to SQLite & ChromaDB.
    """
    logger.info(f"Starting processing for video {video_id} (interval={sampling_interval}, conf={conf_thresh})")
    video = get_video(video_id)
    if not video:
        logger.error(f"Video {video_id} not found in database.")
        return

    file_path = video["file_path"]
    if not os.path.exists(file_path):
        err = f"Video file not found at {file_path}"
        logger.error(err)
        update_video_progress(video_id, progress=0, status="FAILED", error_message=err)
        return

    update_video_progress(video_id, progress=1, status="PROCESSING")

    # Set up video-specific asset directories
    video_frames_dir = config.FRAMES_DIR / video_id
    video_crops_dir = config.CROPS_DIR / video_id
    video_frames_dir.mkdir(parents=True, exist_ok=True)
    video_crops_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        err = "Failed to open video file (codec unsupported or corrupted file)."
        logger.error(err)
        update_video_progress(video_id, progress=0, status="FAILED", error_message=err)
        return

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 25.0  # Fallback default
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = float(total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0

        update_video_metadata(
            video_id=video_id,
            total_frames=total_frames,
            fps=fps,
            duration=duration,
            width=width,
            height=height
        )

        model_mgr = get_model_manager()
        vector_db = get_vector_db()

        frame_idx = 0
        total_detections_count = 0

        pending_crops_pil: List[Image.Image] = []
        pending_detections: List[Dict[str, Any]] = []
        pending_chroma_ids: List[str] = []
        pending_chroma_meta: List[Dict[str, Any]] = []

        def flush_batch():
            nonlocal pending_crops_pil, pending_detections, pending_chroma_ids, pending_chroma_meta
            if not pending_crops_pil:
                return

            try:
                embeddings = model_mgr.encode_crops_batch(pending_crops_pil)
                vector_db.add_batch(
                    ids=pending_chroma_ids,
                    embeddings=embeddings,
                    metadatas=pending_chroma_meta
                )
                save_detections_batch(pending_detections)
            except Exception as e:
                logger.error(f"Error during batch flush: {e}")
                raise e
            finally:
                pending_crops_pil = []
                pending_detections = []
                pending_chroma_ids = []
                pending_chroma_meta = []

        last_progress = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sampling_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = model_mgr.detect_persons(frame_rgb, conf_thresh=conf_thresh)

                if detections:
                    timestamp = float(frame_idx / fps)

                    for box_idx, (x1, y1, x2, y2, conf) in enumerate(detections):
                        crop = frame_rgb[y1:y2, x1:x2]
                        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                            continue

                        # Generate relative asset paths for serving
                        crop_filename = f"crop_{frame_idx:06d}_{box_idx:02d}.jpg"
                        frame_filename = f"frame_{frame_idx:06d}_{box_idx:02d}.jpg"

                        crop_abs_path = video_crops_dir / crop_filename
                        frame_abs_path = video_frames_dir / frame_filename

                        # Save crop image
                        crop_pil = Image.fromarray(crop)
                        crop_pil.save(crop_abs_path, format="JPEG", quality=90)

                        # Draw bounding box and label on annotated frame image
                        annotated_frame = frame.copy()
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
                        label = f"Person {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(
                            annotated_frame,
                            (x1, max(0, y1 - 20)),
                            (x1 + tw + 6, max(20, y1)),
                            (0, 220, 255),
                            -1
                        )
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1 + 3, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA
                        )
                        cv2.imwrite(str(frame_abs_path), annotated_frame)

                        embedding_id = f"{video_id}_{frame_idx:06d}_{box_idx:02d}"
                        rel_frame_path = f"storage/frames/{video_id}/{frame_filename}"
                        rel_crop_path = f"storage/crops/{video_id}/{crop_filename}"

                        detection_dict = {
                            "video_id": video_id,
                            "frame_number": frame_idx,
                            "timestamp": timestamp,
                            "bbox_x1": x1,
                            "bbox_y1": y1,
                            "bbox_x2": x2,
                            "bbox_y2": y2,
                            "confidence": conf,
                            "frame_path": rel_frame_path,
                            "crop_path": rel_crop_path,
                            "embedding_id": embedding_id
                        }

                        chroma_meta = {
                            "video_id": video_id,
                            "frame_number": frame_idx,
                            "timestamp": round(timestamp, 2),
                            "confidence": round(conf, 3),
                            "crop_path": rel_crop_path,
                            "frame_path": rel_frame_path
                        }

                        pending_crops_pil.append(crop_pil)
                        pending_detections.append(detection_dict)
                        pending_chroma_ids.append(embedding_id)
                        pending_chroma_meta.append(chroma_meta)
                        total_detections_count += 1

                        if len(pending_crops_pil) >= config.DEFAULT_EMBEDDING_BATCH_SIZE:
                            flush_batch()

            # Progress calculation
            if total_frames > 0:
                current_progress = int((frame_idx + 1) / total_frames * 95)
            else:
                current_progress = min(90, frame_idx // 10)

            if current_progress > last_progress and (current_progress - last_progress >= 2 or current_progress >= 95):
                last_progress = current_progress
                update_video_progress(
                    video_id=video_id,
                    progress=current_progress,
                    detection_count=total_detections_count
                )

            frame_idx += 1

        # Flush remaining detections
        flush_batch()

        # Update final status
        update_video_progress(
            video_id=video_id,
            progress=100,
            status="COMPLETED",
            detection_count=total_detections_count
        )
        logger.info(f"Video {video_id} processing completed successfully. Total detections: {total_detections_count}")

    except Exception as e:
        logger.exception(f"Unhandled error processing video {video_id}: {e}")
        update_video_progress(
            video_id=video_id,
            progress=0,
            status="FAILED",
            error_message=str(e)
        )
    finally:
        cap.release()


def delete_video_assets(video_id: str):
    """
    Deletes original video file, extracted frames, crops, and vector DB entries.
    """
    logger.info(f"Deleting assets for video {video_id}...")
    # Delete from ChromaDB
    try:
        vector_db = get_vector_db()
        vector_db.delete_by_video_id(video_id)
    except Exception as e:
        logger.warning(f"Error removing ChromaDB embeddings: {e}")

    # Remove storage folders
    video_frames_dir = config.FRAMES_DIR / video_id
    video_crops_dir = config.CROPS_DIR / video_id

    if video_frames_dir.exists():
        shutil.rmtree(video_frames_dir, ignore_errors=True)
    if video_crops_dir.exists():
        shutil.rmtree(video_crops_dir, ignore_errors=True)
