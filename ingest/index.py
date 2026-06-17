"""Build the dense retrieval index: chunk -> embed -> store in pgvector.

Usage:
    python -m ingest.index --limit 25      # cheap dry run over first 25 postings
    python -m ingest.index                 # full corpus
    python -m ingest.index --recreate      # drop & rebuild the chunks table first
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.config import get_settings
from app.db import get_connection
from ingest.chunk import (
    chunk_posting,
    interleave_by_company,
    iter_postings,
    prioritize_chunks,
)
from ingest.embed import embed_texts
from ingest.models import Chunk
from ingest.store import (
    copy_chunks,
    count_chunks,
    create_schema,
    create_vector_index,
    upsert_chunks,
)

DEFAULT_CORPUS = Path("data/raw/postings.jsonl")
STORE_BATCH = 500  # chunks embedded + bulk-copied per round


def plan_chunks(
    corpus: Path, max_chunks: int | None, max_per_posting: int
) -> list[Chunk]:
    """Build a diversified, budgeted chunk set.

    Postings are interleaved across companies for variety; each posting is
    capped to its highest-signal chunks; the total is capped to `max_chunks`
    (used to fit free-tier embedding quotas).
    """
    postings = interleave_by_company(list(iter_postings(corpus)))
    planned: list[Chunk] = []
    for posting in postings:
        planned.extend(prioritize_chunks(chunk_posting(posting), max_per_posting))
        if max_chunks is not None and len(planned) >= max_chunks:
            return planned[:max_chunks]
    return planned


def _flush(conn, settings, batch: list[Chunk], bulk: bool) -> tuple[int, int, float]:
    result = embed_texts([c.content for c in batch], settings)
    writer = copy_chunks if bulk else upsert_chunks
    writer(conn, batch, result.embeddings)
    return len(batch), result.total_tokens, result.estimated_cost(settings.embedding_model)


def build(
    corpus: Path, max_chunks: int | None, max_per_posting: int, recreate: bool
) -> dict:
    settings = get_settings()
    started = time.time()
    written = tokens = 0
    cost = 0.0

    planned = plan_chunks(corpus, max_chunks, max_per_posting)
    print(f"  planned {len(planned)} chunks from {corpus.name}")

    with get_connection(settings) as conn:
        if recreate:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS chunks;")
            conn.commit()
        create_schema(conn, settings)
        # Bulk COPY only when the table started empty; otherwise upsert.
        bulk = count_chunks(conn) == 0

        for start in range(0, len(planned), STORE_BATCH):
            head = planned[start : start + STORE_BATCH]
            n, t, c = _flush(conn, settings, head, bulk)
            written += n; tokens += t; cost += c
            print(f"  embedded+stored {written}/{len(planned)} chunks "
                  f"({time.time() - started:.0f}s)")

        print("  building HNSW vector index...")
        create_vector_index(conn)
        total = count_chunks(conn)

    return {
        "chunks_written": written,
        "chunks_in_table": total,
        "embedding_tokens": tokens,
        "estimated_cost_usd": round(cost, 4),
        "elapsed_s": round(time.time() - started, 1),
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
        "dim": settings.embedding_dim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dense pgvector index")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="cap total chunks embedded; 0 = no cap (all)")
    parser.add_argument("--max-per-posting", type=int, default=0,
                        help="cap high-signal chunks per posting; 0 = no cap")
    parser.add_argument("--recreate", action="store_true", help="drop chunks table first")
    args = parser.parse_args()

    max_chunks = args.max_chunks or None  # 0 -> None (no cap)
    print(f"Indexing {args.corpus} "
          f"(max_chunks={max_chunks}, max_per_posting={args.max_per_posting}, "
          f"recreate={args.recreate})")
    stats = build(args.corpus, max_chunks, args.max_per_posting, args.recreate)
    print("\n=== done ===")
    for k, v in stats.items():
        print(f"  {k:<18} {v}")


if __name__ == "__main__":
    main()
