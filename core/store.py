import faiss
import numpy as np
import json
import os

class FaissStore:
    def __init__(self, dim=384, index_path="faiss_index.bin", metadata_path="metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r") as f:
                metadata_data = json.load(f)
            self.metadata = metadata_data.get("metadata", [])
            self.texts = metadata_data.get("texts", [])
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.texts = []
            self.metadata = []

    def add(self, embedding: np.ndarray, text: str, meta: dict):
        self.index.add(np.array([embedding]))
        self.texts.append(text)
        self.metadata.append(meta)

    def search(self, query_emb: np.ndarray, k=3):
        D, I = self.index.search(np.array([query_emb]), k)
        results = []
        for idx in I[0]:
            if idx < len(self.texts):
                results.append((self.texts[idx], self.metadata[idx]))
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "w") as f:
            json.dump({
                "texts": self.texts,
                "metadata": self.metadata
            }, f)
        print(f"Saved index to {self.index_path} and metadata to {self.metadata_path}")

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            raise FileNotFoundError(f"{self.index_path} not found")

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                metadata_data = json.load(f)
            self.metadata = metadata_data.get("metadata", [])
            self.texts = metadata_data.get("texts", [])
        else:
            raise FileNotFoundError(f"{self.metadata_path} not found")
