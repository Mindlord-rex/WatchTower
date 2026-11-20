import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from src import config

logger = logging.getLogger("watchtower.vector_db")


class VectorDatabase:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorDatabase, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info(f"Connecting to persistent ChromaDB at {config.CHROMA_DIR}...")
        self.client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB initialized. Existing items count: {self.collection.count()}")
        self._initialized = True

    def add_batch(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ):
        """
        Adds embeddings to ChromaDB in chunks below ChromaDB's max batch size limit.
        """
        if not ids:
            return

        batch_size = 2000
        total = len(ids)
        for i in range(0, total, batch_size):
            chunk_ids = ids[i:i + batch_size]
            chunk_embeddings = embeddings[i:i + batch_size]
            chunk_metadatas = metadatas[i:i + batch_size]

            self.collection.add(
                ids=chunk_ids,
                embeddings=chunk_embeddings,
                metadatas=chunk_metadatas
            )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 12,
        video_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Queries ChromaDB for closest embeddings using cosine similarity.
        Optionally filters by video_id.
        """
        where_filter = {"video_id": video_id} if video_id else None

        count = self.collection.count()
        if count == 0:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

        actual_k = min(top_k, count)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            where=where_filter,
            include=["metadatas", "distances"]
        )
        return results

    def delete_by_video_id(self, video_id: str):
        """
        Removes all indexed person embeddings associated with a given video.
        """
        try:
            self.collection.delete(where={"video_id": video_id})
            logger.info(f"Deleted vector embeddings for video {video_id}")
        except Exception as e:
            logger.warning(f"Could not delete vectors by where clause for video {video_id}: {e}")
            # Fallback: query IDs by where or manual check
            try:
                res = self.collection.get(where={"video_id": video_id}, include=[])
                if res and res.get("ids"):
                    self.collection.delete(ids=res["ids"])
            except Exception as e2:
                logger.error(f"Fallback deletion failed: {e2}")

    def count(self) -> int:
        return self.collection.count()


def get_vector_db() -> VectorDatabase:
    return VectorDatabase()
