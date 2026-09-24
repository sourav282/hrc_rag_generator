"""
qa_engine.py
------------
Step 4 of the RAG pipeline: question -> retrieve chunks -> grounded answer.

    ingest(path)   files -> chunks -> embeddings -> vector store
    ask(question)  question -> top-k chunks -> Claude -> answer + sources

The Claude API key is read from the ANTHROPIC_API_KEY environment variable:
    export ANTHROPIC_API_KEY="sk-ant-..."

Install:  pip install anthropic
"""

import os
from dataclasses import dataclass

import anthropic

from src.chunking import Chunk, build_chunks
from src.embedding import Embedder
from src.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Settings (change these to experiment)
# ---------------------------------------------------------------------------
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")  # override via env var
TOP_K = 4            # chunks sent to Claude as context
MIN_SCORE = 0.10     # below this best-match score, skip Claude and say "not found"
MAX_TOKENS = 1024

NOT_FOUND_MESSAGE = "I couldn't find this in the provided documents."

SYSTEM_PROMPT = f"""
        You are a document-grounded assistant. Answer the user's question using **only the information contained in the provided context**.

        ### Rules

        1. **Use only the provided context.** Do not rely on prior knowledge, assumptions, or external information.
        2. **Do not hallucinate.** If the context does not contain enough information to answer the question, clearly say: **"The provided context does not contain enough information to answer this."**
        3. **Ground every factual claim in the context.** Do not introduce facts that cannot be supported by the retrieved documents.
        4. **Cite your sources.** For every factual statement derived from the context, include the relevant source in this format:
        `[source: filename]`
        5. **Use the most relevant source.** If multiple sources support a statement, cite all relevant sources.
        6. **Handle conflicting information carefully.** If sources disagree, explicitly mention the conflict and cite each source rather than choosing one without evidence.
        7. **Distinguish uncertainty.** If the context only partially answers the question, provide the supported information and clearly state what is missing.
        8. **Answer the question directly.** Do not add unnecessary background, assumptions, or explanations.
        9. **Do not mention information retrieval, context windows, embeddings, or internal reasoning unless the user asks about them.**

        ### Response style

        * Concise and direct
        * Clear and factual
        * No unsupported assumptions
        * No external knowledge
        * Include citations inline with the claims they support
        """


@dataclass
class Answer:
    text: str
    sources: list[tuple[Chunk, float]]   # retrieved chunks with similarity scores


def build_context(results: list[tuple[Chunk, float]]) -> str:
    """Format retrieved chunks as numbered passages with their source labels."""
    blocks = []
    for number, (chunk, _score) in enumerate(results, start=1):
        blocks.append(f"[{number}] (source: {chunk.source}, {chunk.location})\n{chunk.text}")
    return "\n\n".join(blocks)


class QAEngine:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        model: str = CLAUDE_MODEL,
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
    ):
        self.store = store if store is not None else VectorStore()
        self.embedder = embedder if embedder is not None else Embedder()
        self.model = model
        self.top_k = top_k
        self.min_score = min_score
        self._client = None  # created lazily so ingest works without an API key

    # -- ingestion ----------------------------------------------------------
    def ingest(self, path: str) -> int:
        """Load, chunk, embed and store a file or folder. Returns chunks added."""
        chunks = build_chunks(path)
        embeddings = self.embedder.embed_texts([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        return len(chunks)

    # -- question answering -------------------------------------------------
    def retrieve(self, question: str) -> list[tuple[Chunk, float]]:
        query_embedding = self.embedder.embed_query(question)
        return self.store.search(query_embedding, self.top_k)

    def ask(self, question: str) -> Answer:
        results = self.retrieve(question)

        # Guardrail: nothing relevant retrieved -> don't ask Claude to guess.
        if not results or results[0][1] < self.min_score:
            return Answer(NOT_FOUND_MESSAGE, results)

        user_prompt = (
            f"Context:\n{build_context(results)}\n\n"
            f"Question: {question}"
        )
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return Answer(text.strip(), results)

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Run: export ANTHROPIC_API_KEY='sk-ant-...'"
                )
            self._client = anthropic.Anthropic()
        return self._client
