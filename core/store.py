import os
import json
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import faiss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaissStore:
    """
    FAISS-based vector store with metadata support.
    """

    def __init__(self, dim: int = 384, store_name: str = "faiss_store"):
        self.dim = dim
        self.store_name = store_name
        self.index_path = f"{self.store_name}_index.faiss"
        self.metadata_path = f"{self.store_name}_metadata.json"

        self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(self.dim)
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.index_to_docstore_id: Dict[int, int] = {}
        self.next_id: int = 0

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load()
        else:
            logger.info("FAISS store not found. Initialized empty store.")

    def add(self, embedding: np.ndarray, text: str, meta: Optional[Dict[str, Any]] = None):
        """
        Add a new embedding with associated text and metadata.

        Args:
            embedding: np.ndarray of shape (dim,) or (n, dim)
            text: associated text
            meta: optional metadata dictionary
        """
        if meta is None:
            meta = {}

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        if embedding.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension {embedding.shape[1]} does not match index dimension {self.dim}")
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)

        self.index.add(embedding)
        self.texts.append(text)
        self.metadata.append(meta)

        start_id = self.index.ntotal - embedding.shape[0]
        for i in range(embedding.shape[0]):
            self.index_to_docstore_id[start_id + i] = self.next_id
            self.next_id += 1

        logger.info(f"Added {embedding.shape[0]} embedding(s) to the store.")

    def search(self, query_emb: np.ndarray, k: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search the FAISS index for nearest neighbors.

        Args:
            query_emb: query embedding
            k: number of neighbors to retrieve

        Returns:
            List of tuples (text, metadata)
        """
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        if query_emb.dtype != np.float32:
            query_emb = query_emb.astype(np.float32)

        distances, indices = self.index.search(query_emb, k)
        results: List[Tuple[str, Dict[str, Any]]] = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append((self.texts[idx], self.metadata[idx]))
        return results

    def save(self):
        """
        Save the FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "texts": self.texts,
                "metadata": self.metadata,
                "index_to_docstore_id": self.index_to_docstore_id,
                "next_id": self.next_id
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved index to {self.index_path} and metadata to {self.metadata_path}")

    def load(self):
        """
        Load the FAISS index and metadata from disk.
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            logger.warning("Index or metadata file not found. Creating empty store.")
            self.index = faiss.IndexFlatL2(self.dim)
            self.texts = []
            self.metadata = []
            self.index_to_docstore_id = {}
            self.next_id = 0
            return

        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.texts = data.get("texts", [])
        self.metadata = data.get("metadata", [])
        self.index_to_docstore_id = data.get("index_to_docstore_id", {})
        self.next_id = data.get("next_id", len(self.texts))
        logger.info(f"Loaded FAISS store with {len(self.texts)} items.")
