"""
chunking.py
-----------
Step 1 of the RAG pipeline: turn raw files into chunks.

    files (PDF / DOCX / CSV)  ->  Segments  ->  fixed-size Chunks

Every chunk carries its source file and a location (page / rows), so answers
can later be grounded and cited.

Install:  pip install pypdf python-docx
Try it:   python src/chunking.py source_docs
"""

import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Settings (change these to experiment)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 150      # words per chunk (~200 tokens; stays under MiniLM's 256-token cap)
CHUNK_OVERLAP = 30    # words shared between consecutive chunks
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv"}

logger = logging.getLogger(__name__)  # warnings print to stderr by default


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """A natural unit of a document (a PDF page, a DOCX body, a group of CSV rows)."""
    text: str
    source: str      # file name
    location: str    # e.g. "page 3", "rows 1-8", "document"


@dataclass
class Chunk:
    """A fixed-size piece of text that will be embedded and stored."""
    text: str
    source: str
    location: str


# ---------------------------------------------------------------------------
# Loaders: one function per file type, each returns a list of Segments
# ---------------------------------------------------------------------------
def load_pdf(path: Path) -> list[Segment]:
    """One segment per page (keeps page numbers for citations)."""
    reader = PdfReader(str(path))
    segments = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as error:  # one bad page shouldn't lose the whole PDF
            logger.warning("%s page %d unreadable (%s)", path.name, page_number, error)
            continue
        if text:
            segments.append(Segment(text, path.name, f"page {page_number}"))
    return segments


def load_docx(path: Path) -> list[Segment]:
    """Paragraphs first, then table rows (cells joined with ' | ')."""
    doc = DocxDocument(str(path))
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    text = "\n".join(parts)
    return [Segment(text, path.name, "document")] if text else []


def load_csv(path: Path, max_words: int = CHUNK_SIZE) -> list[Segment]:
    """
    Turn each row into 'column: value | column: value' text, then pack whole
    rows into segments of up to `max_words` words. Rows are never split
    across segments (unless a single row is longer than max_words).
    Row numbers count data rows only (the header is not counted).
    """
    segments = []
    buffer, buffer_words = [], 0
    first_row, last_row = 1, 0

    def flush():
        segments.append(
            Segment("\n".join(buffer), path.name, f"rows {first_row}-{last_row}")
        )

    # errors="replace": bad bytes become '?' instead of crashing the load
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row_number, row in enumerate(csv.DictReader(f), start=1):
            line = " | ".join(f"{k}: {v}" for k, v in row.items() if k and v)
            if not line:
                continue
            line_words = len(line.split())
            if buffer and buffer_words + line_words > max_words:
                flush()
                buffer, buffer_words, first_row = [], 0, row_number
            buffer.append(line)
            buffer_words += line_words
            last_row = row_number
        if buffer:
            flush()
    return segments


# Add a new file type by writing a loader above and registering it here.
LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".csv": load_csv,
}


def load_documents(path: str | Path) -> list[Segment]:
    """
    Load a single file, or every supported file inside a folder (recursive).
    Blank, unreadable or corrupted files are skipped with a warning instead of
    stopping the whole run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            raise ValueError(
                f"No supported files {sorted(SUPPORTED_EXTENSIONS)} found in {path}"
            )
    elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
        files = [path]
    else:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    segments = []
    for file in files:
        try:
            file_segments = LOADERS[file.suffix.lower()](file)
        except Exception as error:  # corrupted, encrypted, wrong format, etc.
            logger.warning("Skipping %s: could not read file (%s)", file.name, error)
            continue
        if not file_segments:
            logger.warning(
                "Skipping %s: no extractable text (blank or scanned file?)", file.name
            )
            continue
        segments.extend(file_segments)
    return segments


# ---------------------------------------------------------------------------
# Chunking: fixed-size word windows with overlap
# ---------------------------------------------------------------------------
def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Slide a window of `chunk_size` words, moving `chunk_size - overlap` each step."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return pieces


def chunk_segments(
    segments: list[Segment],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split every segment into fixed-size chunks, carrying metadata along."""
    chunks = []
    for segment in segments:
        pieces = split_text(segment.text, chunk_size, overlap)
        for i, piece in enumerate(pieces, start=1):
            location = segment.location
            if len(pieces) > 1:
                location += f" (part {i}/{len(pieces)})"
            chunks.append(Chunk(piece, segment.source, location))
    return chunks


def build_chunks(
    path: str | Path,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Main entry point: path (file or folder) -> list of Chunks."""
    segments = load_documents(path)
    chunks = chunk_segments(segments, chunk_size, overlap)
    if not chunks:
        raise ValueError(f"No text could be extracted from '{path}'")
    return chunks


# ---------------------------------------------------------------------------
# Quick manual check:  python src/chunking.py <file-or-folder>
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "source_docs"
    all_chunks = build_chunks(target)
    print(f"Built {len(all_chunks)} chunks from '{target}'\n")
    for chunk in all_chunks[:5]:
        preview = chunk.text[:120].replace("\n", " ")
        print(f"[{chunk.source} | {chunk.location}] {preview}...")
