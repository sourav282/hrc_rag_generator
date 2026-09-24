"""
main.py  -  command line interface for the RAG Generator

    python main.py ingest source_docs              # build an index from a file or folder
    python main.py ask "your question"             # one-off question
    python main.py chat                            # interactive Q&A loop

    --index-dir <folder>   where the index is saved/loaded (default: index).
                           Use a different folder per document set.
"""

import argparse
import logging
import sys

from src.qa_engine import QAEngine
from src.vector_store import VectorStore

DEFAULT_INDEX_DIR = "index"


def print_answer(answer) -> None:
    print(f"\n{answer.text}\n")
    if answer.sources:
        print("Retrieved sources:")
        for number, (chunk, score) in enumerate(answer.sources, start=1):
            print(f"  [{number}] {chunk.source} | {chunk.location} | score {score:.2f}")
        print()


def cmd_ingest(args) -> None:
    engine = QAEngine()                       # fresh, empty store = a new document set
    count = engine.ingest(args.path)
    engine.store.save(args.index_dir)
    print(f"Ingested {count} chunks from '{args.path}' -> saved to '{args.index_dir}/'")


def load_engine(args) -> QAEngine:
    return QAEngine(store=VectorStore.load(args.index_dir), top_k=args.top_k)


def cmd_ask(args) -> None:
    print_answer(load_engine(args).ask(args.question))


def cmd_chat(args) -> None:
    engine = load_engine(args)
    print(f"Loaded {len(engine.store)} chunks from '{args.index_dir}/'. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            print_answer(engine.ask(question))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG Generator")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="build an index from a file or folder")
    p.add_argument("path")
    p.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask a single question")
    p.add_argument("question")
    p.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    p.add_argument("--top-k", type=int, default=4)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive question loop")
    p.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    p.add_argument("--top-k", type=int, default=4)
    p.set_defaults(func=cmd_chat)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()
