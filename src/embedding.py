"""
embedding.py
------------
Step 2 of the RAG pipeline: turn text into vectors with all-MiniLM-L6-v2.

Vectors are L2-normalised, so a plain dot product between two of them equals
their cosine similarity (the vector store relies on this).

Install:  pip install sentence-transformers
"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, max 256 word pieces per input
BATCH_SIZE = 32


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        # Downloads the model on first use, then loads it from the local cache.
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Embed many texts -> array of shape (len(texts), 384)."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one question -> array of shape (384,)."""
        return self.embed_texts([query])[0]
