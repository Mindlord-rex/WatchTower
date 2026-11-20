import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src import config

logger = logging.getLogger("watchtower.database")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.DATABASE_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'UPLOADED',
            progress INTEGER DEFAULT 0,
            total_frames INTEGER DEFAULT 0,
            fps REAL DEFAULT 0.0,
            duration REAL DEFAULT 0.0,
            width INTEGER DEFAULT 0,
            height INTEGER DEFAULT 0,
            sampling_interval INTEGER DEFAULT 30,
            detection_count INTEGER DEFAULT 0,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            frame_number INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            bbox_x1 INTEGER NOT NULL,
            bbox_y1 INTEGER NOT NULL,
            bbox_x2 INTEGER NOT NULL,
            bbox_y2 INTEGER NOT NULL,
            confidence REAL NOT NULL,
            frame_path TEXT NOT NULL,
            crop_path TEXT NOT NULL,
            embedding_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
        )
    ''')

    c.execute("CREATE INDEX IF NOT EXISTS idx_detections_video_id ON detections(video_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_detections_embedding_id ON detections(embedding_id)")

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")


def create_video_record(
    video_id: str,
    filename: str,
    file_path: str,
    sampling_interval: int = config.DEFAULT_FRAME_INTERVAL,
    total_frames: int = 0,
    fps: float = 0.0,
    duration: float = 0.0,
    width: int = 0,
    height: int = 0
) -> Dict[str, Any]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO videos (
            id, filename, file_path, status, progress, total_frames,
            fps, duration, width, height, sampling_interval, detection_count
        ) VALUES (?, ?, ?, 'UPLOADED', 0, ?, ?, ?, ?, ?, ?, 0)
    ''', (video_id, filename, file_path, total_frames, fps, duration, width, height, sampling_interval))
    conn.commit()
    conn.close()
    return get_video(video_id)


def update_video_progress(
    video_id: str,
    progress: int,
    status: Optional[str] = None,
    detection_count: Optional[int] = None,
    error_message: Optional[str] = None
):
    conn = get_db_connection()
    c = conn.cursor()
    updates = ["progress = ?"]
    params: List[Any] = [progress]

    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if detection_count is not None:
        updates.append("detection_count = ?")
        params.append(detection_count)
    if error_message is not None:
        updates.append("error_message = ?")
        params.append(error_message)

    params.append(video_id)
    query = f"UPDATE videos SET {', '.join(updates)} WHERE id = ?"
    c.execute(query, params)
    conn.commit()
    conn.close()


def update_video_metadata(
    video_id: str,
    total_frames: int,
    fps: float,
    duration: float,
    width: int,
    height: int
):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        UPDATE videos
        SET total_frames = ?, fps = ?, duration = ?, width = ?, height = ?
        WHERE id = ?
    ''', (total_frames, fps, duration, width, height, video_id))
    conn.commit()
    conn.close()


def get_video(video_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def list_videos() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM videos ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_video_record(video_id: str) -> Optional[Dict[str, Any]]:
    video = get_video(video_id)
    if not video:
        return None
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM detections WHERE video_id = ?", (video_id,))
    c.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    conn.commit()
    conn.close()
    return video


def save_detections_batch(detections: List[Dict[str, Any]]):
    if not detections:
        return
    conn = get_db_connection()
    c = conn.cursor()
    c.executemany('''
        INSERT INTO detections (
            video_id, frame_number, timestamp,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            confidence, frame_path, crop_path, embedding_id
        ) VALUES (
            :video_id, :frame_number, :timestamp,
            :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2,
            :confidence, :frame_path, :crop_path, :embedding_id
        )
    ''', detections)
    conn.commit()
    conn.close()


def get_detections_by_embedding_ids(embedding_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not embedding_ids:
        return {}
    conn = get_db_connection()
    c = conn.cursor()
    placeholders = ",".join("?" for _ in embedding_ids)
    query = f"""
        SELECT d.*, v.filename AS video_filename, v.file_path AS video_file_path
        FROM detections d
        JOIN videos v ON d.video_id = v.id
        WHERE d.embedding_id IN ({placeholders})
    """
    c.execute(query, embedding_ids)
    rows = c.fetchall()
    conn.close()
    return {row["embedding_id"]: dict(row) for row in rows}


def get_dashboard_stats() -> Dict[str, Any]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM videos")
    total_videos = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM detections")
    total_detections = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM videos WHERE status = 'COMPLETED'")
    completed_videos = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM videos WHERE status = 'PROCESSING'")
    processing_videos = c.fetchone()[0]

    c.execute("SELECT * FROM videos ORDER BY created_at DESC LIMIT 5")
    recent_videos = [dict(r) for r in c.fetchall()]

    conn.close()
    return {
        "total_videos": total_videos,
        "total_detections": total_detections,
        "completed_videos": completed_videos,
        "processing_videos": processing_videos,
        "recent_videos": recent_videos
    }


# Automatically initialize tables on import
init_db()
