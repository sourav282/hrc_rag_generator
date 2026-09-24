# hrc_rag_generator

A small, modular command-line tool that builds a Retrieval-Augmented Generation (RAG) app over whatever documents you give it at runtime, then answers questions grounded in those documents, with citations.

- **Runtime ingestion** of PDF, DOCX and CSV files (a single file or a whole folder)
- **Grounded answers** from Claude, citing the passages used, with an explicit "not found" behaviour
- **Any document set, no code changes:** each set lives in its own index folder
- **Modular:** loading/chunking, embedding, storage and answering are separate files you can change independently

## Project structure

```
rag_generator/
├── main.py              # CLI entry point: ingest / ask / chat
├── requirements.txt
├── src/
│   ├── chunking.py      # PDF / DOCX / CSV loaders + fixed-size chunking
│   ├── embedding.py     # all-MiniLM-L6-v2 embeddings
│   ├── vector_store.py  # in-memory NumPy vector store, save/load to disk
│   └── qa_engine.py     # retrieval + grounded answer generation with Claude
└── source_docs/         # sample documents to try it on
    ├── leave_policy.docx
    ├── nimbus_thermostat_faq.pdf
    └── orders.csv
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."      # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

Optional: choose a different Claude model without editing code.

```bash
export CLAUDE_MODEL="claude-sonnet-5"      # this is the default
```

The first `ingest` downloads the `all-MiniLM-L6-v2` model (roughly 90 MB) and caches it locally.

## Usage

**1. Ingest documents** (builds and saves an index):

```bash
python main.py ingest source_docs
python main.py ingest path/to/report.pdf --index-dir index_reports
```

**2. Ask questions:**

```bash
python main.py ask "How many days of annual leave do I get?"
python main.py chat                          # interactive loop, type 'exit' to quit
```

### Grounding behaviour

- Claude is told to use only the provided passages, cite them, and reply *"I couldn't find this in the provided documents."* when the answer isn't there.
- If even the best-matching chunk scores below `MIN_SCORE`, Claude is not called at all and the not-found message is returned.
- Document text is treated as untrusted data, not as instructions.

### Handling bad files

Blank, corrupted, encrypted or otherwise unreadable files are skipped with a warning naming the file and the reason; the rest of the folder still loads. A single unreadable PDF page only skips that page. If nothing can be extracted at all, `ingest` stops with a clear error instead of building an empty index.

## Configuration

Settings sit at the top of each file.

| Setting | File | Default | Notes |
|---|---|---|---|
| `CHUNK_SIZE` | `chunking.py` | 150 words | Keep it under about 190 words to respect MiniLM's 256-token limit. |
| `CHUNK_OVERLAP` | `chunking.py` | 30 words | Must be smaller than `CHUNK_SIZE`. Not applied to CSV chunks. |
| `MODEL_NAME` | `embedding.py` | `all-MiniLM-L6-v2` | Any sentence-transformers model works. |
| `CLAUDE_MODEL` | `qa_engine.py` | `claude-sonnet-5` | Overridable through the environment variable. |
| `TOP_K` | `qa_engine.py` | 4 | Chunks sent to Claude. |
| `MIN_SCORE` | `qa_engine.py` | 0.10 | Cutoff below which the answer is "not found". Tune after trying your own documents. |
| `MAX_TOKENS` | `qa_engine.py` | 1024 | Maximum answer length. |


## Sample questions for `source_docs/`

- How many days of annual leave do full-time employees get?
- How many unused annual leave days can be carried forward, and when do they expire?
- When is a medical certificate required for sick leave?
- Does the thermostat support 5 GHz Wi-Fi?
- How do I factory reset the Nimbus thermostat?
- What is the warranty period, and does it cover incorrect installation?
- What is the status of order ORD-1003?
- What is the company's remote work policy? *(not in the documents, so it should answer "not found")*

