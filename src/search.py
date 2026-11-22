import logging
from typing import List, Dict, Any, Optional

from src.models import get_model_manager
from src.vector_db import get_vector_db
from src.database import get_detections_by_embedding_ids

logger = logging.getLogger("watchtower.search")


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    ms = int((seconds - int(seconds)) * 10)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms}"
    return f"{m:02d}:{s:02d}.{ms}"


def semantic_person_search(
    query_text: str,
    top_k: int = 12,
    video_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Performs semantic search across indexed person detections using natural language query.
    Uses CLIP text encoder and ChromaDB cosine similarity search.
    """
    clean_query = query_text.strip()
    if not clean_query:
        return []

    logger.info(f"Executing semantic search for: '{clean_query}', top_k={top_k}, video_id={video_id}")
    model_mgr = get_model_manager()
    vector_db = get_vector_db()

    query_embedding = model_mgr.encode_text_query(clean_query)
    results = vector_db.search(
        query_embedding=query_embedding,
        top_k=top_k,
        video_id=video_id
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not ids:
        return []

    # Fetch detailed detection and video records from SQLite
    db_records = get_detections_by_embedding_ids(ids)

    enriched_results: List[Dict[str, Any]] = []
    for rank, (emb_id, dist, meta) in enumerate(zip(ids, distances, metadatas)):
        db_rec = db_records.get(emb_id, {})
        # Cosine distance to similarity: cosine similarity = 1 - cosine distance
        similarity = max(0.0, min(1.0, 1.0 - dist))
        similarity_pct = int(similarity * 100)

        ts = meta.get("timestamp", db_rec.get("timestamp", 0.0))
        formatted_ts = format_timestamp(ts)

        vid_id = meta.get("video_id", db_rec.get("video_id", ""))
        vid_name = db_rec.get("video_filename", f"Video {vid_id[:8]}")

        frame_path = db_rec.get("frame_path", meta.get("frame_path", ""))
        crop_path = db_rec.get("crop_path", meta.get("crop_path", ""))

        x1 = db_rec.get("bbox_x1", 0)
        y1 = db_rec.get("bbox_y1", 0)
        x2 = db_rec.get("bbox_x2", 0)
        y2 = db_rec.get("bbox_y2", 0)

        enriched_results.append({
            "rank": rank + 1,
            "embedding_id": emb_id,
            "video_id": vid_id,
            "video_filename": vid_name,
            "timestamp": ts,
            "timestamp_formatted": formatted_ts,
            "frame_number": meta.get("frame_number", db_rec.get("frame_number", 0)),
            "confidence": db_rec.get("confidence", meta.get("confidence", 0.0)),
            "similarity": similarity,
            "similarity_pct": similarity_pct,
            "distance": dist,
            "frame_path": frame_path,
            "crop_path": crop_path,
            "bbox": [x1, y1, x2, y2]
        })

    return enriched_results
