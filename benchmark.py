
#!/usr/bin/env python3
"""Benchmark indexed lyrics attribution against a naive full scan."""

import argparse
import random
import time

try:
    from engine import AttributionEngine
    from engine.normalize import tokenize, token_texts
    from engine.similarity import lcs_ratio
except ImportError as exc:
    raise SystemExit(
        f"Import error: {exc}\n"
        "Run this from the project root (the folder that contains "
        "the engine/ package)."
    ) from exc


WORDS = (
    "river stone window rain paper street light tower engine orchard "
    "signal morning chorus refrain garden quarter tin roof salt tide "
    "highway exit coffee metronome silence bruise ticket platform "
    "shadow ladder candle harbour winter letter"
).split()

QUERY = (
    "so carry me home through the blue and the rain\n"
    "i am half a chorus you are the refrain\n"
    "a line that belongs to nobody at all"
)

SOURCE_LYRICS = (
    "so carry me home through the blue and the rain\n"
    "i am half a chorus you are the refrain\n"
    "if the morning forgets us let it forget\n"
    "we were louder than anything the quiet allowed yet"
)


def random_song(rng, lines=12, per_line=8):
    """Generate random lyrics for benchmarking."""
    return "\n".join(
        " ".join(rng.choice(WORDS) for _ in range(per_line))
        for _ in range(lines)
    )


def build(n, seed=7):
    """Build a synthetic catalogue with one known source song."""
    if n < 1:
        raise ValueError("The number of songs must be at least 1.")

    rng = random.Random(seed)

    records = [
        {
            "id": f"G{i:05d}",
            "title": f"Generated {i}",
            "artist": "synthetic",
            "lyrics": random_song(rng),
        }
        for i in range(n)
    ]

    source_index = n // 2
    records[source_index]["lyrics"] = SOURCE_LYRICS
    records[source_index]["title"] = "Planted Source"

    return records


def naive_scan(records, query, top=3):
    """Compare the query against every song using LCS."""
    query_tokens = token_texts(tokenize(query))
    scored = []

    for record in records:
        song_tokens = token_texts(tokenize(record["lyrics"]))
        score = lcs_ratio(query_tokens, song_tokens)
        scored.append((score, record["title"]))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[:top]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark indexed lyrics attribution."
    )
    parser.add_argument(
        "--songs",
        type=int,
        default=400,
        help="Number of synthetic songs (default: 400).",
    )
    args = parser.parse_args()

    if args.songs < 1:
        parser.error("--songs must be at least 1.")

    try:
        records = build(args.songs)

        # Build the index.
        start = time.perf_counter()
        engine = AttributionEngine(k=5, top_k=3)
        engine.ingest_records(records)
        build_time = time.perf_counter() - start

        # Search using the index.
        start = time.perf_counter()
        report = engine.attribute(QUERY)
        indexed_time = time.perf_counter() - start

        # Search every song without using the index.
        start = time.perf_counter()
        naive_top = naive_scan(records, QUERY)
        naive_time = time.perf_counter() - start

        print("\n=== Lyrics Similarity Benchmark ===")
        print(f"Catalogue size      : {args.songs} songs")
        print(f"Index build (once)  : {build_time * 1000:8.2f} ms")
        print(f"Indexed query       : {indexed_time * 1000:8.2f} ms")
        print(f"Naive full scan     : {naive_time * 1000:8.2f} ms")

        if indexed_time > 0:
            speedup = naive_time / indexed_time
            print(f"Speedup             : {speedup:8.2f}x")

        print("\n=== Results ===")

        if report.best:
            print(f"Indexed top hit     : {report.best.song.title}")
            print(f"Indexed score       : {report.best.score:.3f}")
        else:
            print("Indexed top hit     : No match found")

        if naive_top:
            print(f"Naive top hit       : {naive_top[0][1]}")
            print(f"Naive LCS score     : {naive_top[0][0]:.3f}")
        else:
            print("Naive top hit       : No match found")

        print("\nBenchmark completed successfully.")

    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise SystemExit(f"Benchmark failed: {exc}") from exc


if __name__ == "__main__":
    main()
