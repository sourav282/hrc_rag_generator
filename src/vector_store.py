"""
vector_store.py
---------------
Step 3 of the RAG pipeline: keep chunk vectors and find the closest ones.

A deliberately simple in-memory store (NumPy). Because the embeddings are
normalised, similarity search is just a matrix product. The whole store can be
saved to / loaded from a folder, so a document set only needs to be embedded once.

Swap in FAISS / Chroma later by keeping the same four methods:
add(), search(), save(), load().
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.chunking import Chunk

EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.json"


class VectorStore:
    def __init__(self):
        self.embeddings: np.ndarray | None = None   # shape (n_chunks, dim)
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[tuple[Chunk, float]]:
        """Return the top_k most similar chunks as (chunk, cosine_score), best first."""
        if self.embeddings is None or not self.chunks:
            return []
        scores = self.embeddings @ query_embedding
        top_k = min(top_k, len(self.chunks))
        best = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in best]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / EMBEDDINGS_FILE, self.embeddings)
        with open(directory / CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        emb_path, chunks_path = directory / EMBEDDINGS_FILE, directory / CHUNKS_FILE
        if not emb_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No saved index in '{directory}'. Run the 'ingest' command first."
            )
        store = cls()
        store.embeddings = np.load(emb_path)
        with open(chunks_path, encoding="utf-8") as f:
            store.chunks = [Chunk(**item) for item in json.load(f)]
        return store
